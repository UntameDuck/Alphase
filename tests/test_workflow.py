import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from pypdf import PdfWriter
from alphase.config import DEFAULT
from alphase.data import PALETTE,save_json,inspect,extract_archives,sha256
from alphase.engine import train,Predictor,evaluate,infer,atomic_save
from alphase.ensemble import search
from alphase.experiments import plan,compare,run
from alphase.submission import validate_predictions,build_package,reproducibility_check,verify_package,output_name,load_contract
from alphase.reporting import draft
from alphase.budget import estimate
from fixtures import metric_record,write_cache

def fixture(root):
    rng=np.random.default_rng(42)
    for split,n in [('train',2),('valid',2),('test',1)]:
        for folder in ['images','masks']:(root/split/folder).mkdir(parents=True)
        for i in range(n):
            shape=(45+i,53+i);x=rng.integers(0,255,(*shape,3),dtype=np.uint8);y=rng.integers(0,4,shape,dtype=np.uint8)
            Image.fromarray(x).save(root/split/'images'/f'{split}{i}_image.png')
            if split!='test':Image.fromarray(PALETTE[y]).save(root/split/'masks'/f'{split}{i}_mask.png')

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(Path(d)/'train.zip','w') as z:z.writestr('../outside','bad')
            with self.assertRaises(ValueError):extract_archives(d,Path(d)/'out')
    def test_filename_mapping(self):
        c=load_contract();c['filename_map']={'a_image.png':'official.png'};self.assertEqual(output_name('a_image.png',c),'official.png')
        c['filename_map']['a_image.png']='../bad.png'
        with self.assertRaises(ValueError):output_name('a_image.png',c)
    def test_plan_budget_and_empty_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=plan(DEFAULT,d,[42,43],['A','B','C','D','E','F','G']);self.assertEqual(len(p['runs']),14)
            for run in p['runs']:
                cfg=json.loads((d/run['config']).read_text(encoding='utf-8'));self.assertEqual(cfg['patch_size'],DEFAULT['patch_size'])
            r=compare(d/'plan.json',d/'not_run',d/'comparison');self.assertIsNone(r['suggested_variant']);self.assertEqual(len(r['missing']),14)
            with self.assertRaises(ValueError):plan(DEFAULT,d/'pre',[42],['Pretrained'])
    def test_ensemble_search_known_optimum(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);y=np.array([[0,1],[2,3]],np.uint8);good=np.eye(4,dtype=np.float32)[y].transpose(2,0,1);bad=np.roll(good,1,axis=0)
            for name,p in [('good',good),('bad',bad)]:
                write_cache(d/name,p,y,DEFAULT)
            r=search([d/'good',d/'bad'],d/'search.json');self.assertEqual(r['best']['miou'],1);self.assertEqual(r['best']['weights'],[1.,0.])
            np.savez_compressed(d/'bad'/'x_image.npz',probabilities=bad,target=np.zeros_like(y))
            with self.assertRaises(ValueError):search([d/'good',d/'bad'],d/'bad.json')
    def test_factor_effects_and_edge_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);plan(DEFAULT,d/'plan',[42],['A','B','C','D','E','F','G'])
            for ident,score in {'A':.4,'B':.5,'C':.55,'D':.65,'E':.7,'F':.68,'G':.72}.items():
                m=metric_record(json.loads((d/'plan'/f'{ident}_s42.json').read_text(encoding='utf-8')),score)
                save_json(d/'runs'/f'{ident}_s42'/'evaluation'/'metrics.json',m)
            r=compare(d/'plan'/'plan.json',d/'runs',d/'comparison');self.assertEqual(r['suggested_variant'],'E')
            self.assertAlmostEqual(r['factor_effects'][0]['interaction'],0.)
            self.assertFalse(next(g for g in r['selection_gates'] if g['child']=='G')['eligible'])

    def test_runner_completed_smoke_is_idempotent_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);fixture(d/'data')
            cfg=DEFAULT|{'patch_size':64,'overlap':16,'batch_size':2,'patches_per_image':1,'epochs':1,'device':'cpu'}
            plan(cfg,d/'plan',[42],['A'])
            first=run(d/'plan/plan.json',d/'data',d/'runs',steps=1)
            files=[d/'runs/A_s42'/name for name in ['last.pt','best.pt','evaluation/metrics.json']]
            before=[sha256(p) for p in files]
            second=run(d/'plan/plan.json',d/'data',d/'runs',steps=1)
            self.assertEqual(before,[sha256(p) for p in files])
            self.assertEqual(first['runs'],second['runs']);self.assertIsNone(second['suggested_variant'])
            self.assertIn('smoke_incomplete_or_untracked_training',second['exclusions']['A'])
    def test_end_to_end_resume_ensemble_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);root=d/'data';fixture(root);self.assertEqual(inspect(root)['splits']['train']['count'],2)
            cfg=DEFAULT|{'architecture':'unet','encoder':'resnet18','patch_size':64,'overlap':16,'batch_size':2,'patches_per_image':3,'epochs':2,'device':'cpu',
                'ema_decay':.99,'accumulation_steps':2,'sampler':'phase_boundary','dice_reduction':'image_present'}
            out=d/'run'
            def capture(path,state):
                atomic_save(path,state)
                if Path(path).name=='last.pt' and state['epoch']==0:atomic_save(out/'epoch0.pt',state)
            with patch('alphase.engine.atomic_save',side_effect=capture):train(root,cfg,out)
            train(root,cfg,d/'resumed',resume=out/'epoch0.pt')
            original=torch.load(out/'last.pt',weights_only=False);resumed=torch.load(d/'resumed'/'last.pt',weights_only=False)
            for k,v in original['model'].items():torch.testing.assert_close(v,resumed['model'][k],rtol=0,atol=0)
            for key in ['eval_model','optimizer','torch_rng']:torch.testing.assert_close(original[key],resumed[key],rtol=0,atol=0)
            self.assertEqual(original['scheduler'],resumed['scheduler']);self.assertEqual(original['ema_updates'],4)
            self.assertEqual(original['ema_updates'],resumed['ema_updates'])
            self.assertEqual([r['optimizer_steps'] for r in original['history']],[2,2])
            cp=[str(out/'best.pt')];predictor=Predictor(cp,'cpu',tta=True);r=evaluate(predictor,root,d/'evaluation',cache=True)
            self.assertEqual(r['image_count'],2);self.assertTrue((d/'evaluation'/'report.html').exists())
            cost=estimate(out,.432,d/'budget.json');self.assertEqual(cost['remaining_max_epochs'],0)
            save_json(d/'data_report.json',inspect(root));save_json(d/'comparison.json',{'runs':[],'missing':['A']})
            draft(d/'data_report.json',d/'evaluation'/'metrics.json',d/'comparison.json',d/'report_draft.md')
            self.assertIn('Missing experiments: A',(d/'report_draft.md').read_text(encoding='utf-8'))
            ensemble=evaluate(Predictor(cp+cp,'cpu',[.25,.75],True),root);self.assertAlmostEqual(r['image_macro_miou'],ensemble['image_macro_miou'])
            contract=load_contract();contract.update(expected_count=1,output_format_confirmed=True,absent_policy_confirmed=True)
            save_json(d/'contract.json',contract);infer(root,cp,d/'pred','cpu',tta=True,contract=d/'contract.json')
            self.assertEqual(validate_predictions(root,d/'pred',contract)['count'],1)
            self.assertTrue(reproducibility_check(root,cp,d/'pred','cpu',tta=True,contract=d/'contract.json')['identical_png_hashes'])
            writer=PdfWriter();writer.add_blank_page(width=100,height=100);writer.write(d/'report.pdf')
            with self.assertRaisesRegex(ValueError,'manifest'):
                build_package(root,d/'pred',cp,d/'report.pdf','BAD_TTA',d/'BAD_TTA.zip',d/'contract.json',tta=False)
            package=build_package(root,d/'pred',cp,d/'report.pdf','TEST',d/'TEST.zip',d/'contract.json',tta=True)
            self.assertEqual(package['report_pages'],1)
            self.assertTrue(package['repeated_inference_verified'])
            with zipfile.ZipFile(d/'TEST.zip') as z:
                self.assertFalse(any('/masks/' in n or 'last.pt' in n for n in z.namelist()))
            interpreter=os.environ.get('ALPHASE_VERIFY_PYTHON')
            if interpreter:self.assertEqual(verify_package(d/'TEST.zip',root,interpreter)['status'],'pass')
            broken=cfg|{'lr':.01}
            with self.assertRaises(ValueError):train(root,broken,d/'bad_resume',resume=out/'last.pt',steps=1)
            image=np.asarray(Image.open(d/'pred/test0_mask.png').convert('RGB')).copy()
            image[0,0]=PALETTE[1] if np.array_equal(image[0,0],PALETTE[0]) else PALETTE[0]
            Image.fromarray(image).save(d/'pred/test0_mask.png')
            with self.assertRaisesRegex(ValueError,'hashes changed'):
                build_package(root,d/'pred',cp,d/'report.pdf','TAMPERED',d/'TAMPERED.zip',d/'contract.json',tta=True)
            Image.fromarray(np.zeros((45,53,3),np.uint8)).save(d/'pred'/'test0_mask.png')
            with self.assertRaises(ValueError):validate_predictions(root,d/'pred',contract)

if __name__=='__main__':unittest.main()
