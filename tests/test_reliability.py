import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import Dataset
from alphase.config import DEFAULT,load_config
from alphase.data import save_json,sha256
from alphase.engine import train,atomic_save,evaluate
from alphase.experiments import plan,compare,run
from alphase.ensemble import search
from alphase.provenance import prediction_fingerprint
from alphase.submission import load_contract
from alphase.budget import estimate
from fixtures import metric_record,write_cache


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Conv2d(3,4,1);self.head=torch.nn.Conv2d(4,4,1)
    def forward(self,x):return self.head(self.encoder(x))


class TinyDataset(Dataset):
    def __init__(self,*args):pass
    def __len__(self):return 5
    def set_epoch(self,epoch):pass
    def __getitem__(self,index):return torch.full((3,32,32),index/5),torch.zeros((32,32),dtype=torch.long)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def training_context(self):
        stack=contextlib.ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch('alphase.engine.build_model',side_effect=lambda *a,**kw:TinyModel()))
        stack.enter_context(patch('alphase.engine.PatchDataset',TinyDataset))
        stack.enter_context(patch('alphase.engine.data_fingerprint',return_value='synthetic'))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return DEFAULT|{'device':'cpu','patch_size':32,'overlap':8,'epochs':5,'patience':1,'batch_size':2,'ema_decay':.9,'accumulation_steps':2,'dice_reduction':'image_present'}

    def test_completed_resume_is_noop_and_fork_is_self_contained(self):
        cfg=self.training_context();out=self.root/'run'
        with patch('alphase.engine.evaluate',return_value={'image_macro_miou':.5,'class_macro_iou':[.5]*4}):
            train('synthetic',cfg,out)
            first=torch.load(out/'last.pt',weights_only=False);before=sha256(out/'best.pt')
            train('synthetic',cfg,out,resume=out/'last.pt')
            second=torch.load(out/'last.pt',weights_only=False)
            self.assertEqual(first['epoch'],1);self.assertEqual(second['epoch'],1)
            self.assertEqual(before,sha256(out/'best.pt'))
            train('synthetic',cfg,self.root/'fork',resume=out/'last.pt')
            for name in ['best.pt','last.pt','inference.pt','history.csv','run_status.json']:self.assertTrue((self.root/'fork'/name).exists())
            self.assertEqual(len(second['history']),2)
            with self.assertRaisesRegex(ValueError,'step budget'):train('synthetic',cfg,out,resume=out/'last.pt',steps=1)

    def test_crash_after_commit_recovers_best_and_history(self):
        cfg=self.training_context()|{'epochs':2,'patience':2};out=self.root/'run'
        count=0
        def crash(path,state):
            nonlocal count
            if Path(path).name=='best.pt':
                count+=1
                if count==2:raise RuntimeError('simulated interrupted best export')
            atomic_save(path,state)
        with patch('alphase.engine.evaluate',side_effect=[{'image_macro_miou':s,'class_macro_iou':[s]*4} for s in [.5,.6]]),patch('alphase.engine.atomic_save',side_effect=crash):
            with self.assertRaises(RuntimeError):train('synthetic',cfg,out)
        train('synthetic',cfg,out,resume=out/'last.pt')
        last=torch.load(out/'last.pt',weights_only=False);best=torch.load(out/'best.pt',weights_only=False)
        self.assertEqual(best['training_run']['best_epoch'],2)
        torch.testing.assert_close(best['model'],last['best_eval_model'],rtol=0,atol=0)
        self.assertEqual(len((out/'history.csv').read_text().splitlines()),3)

    def test_resume_does_not_overwrite_unrelated_destination(self):
        cfg=self.training_context();out=self.root/'run'
        with patch('alphase.engine.evaluate',return_value={'image_macro_miou':.5,'class_macro_iou':[.5]*4}):train('synthetic',cfg,out)
        save_json(self.root/'other/config.json',{'user':'preserve'})
        before=sha256(self.root/'other/config.json')
        with self.assertRaises(ValueError):train('synthetic',cfg,self.root/'other',resume=out/'last.pt')
        self.assertEqual(before,sha256(self.root/'other/config.json'))

    def test_corrupt_derived_checkpoint_is_recovered(self):
        cfg=self.training_context();out=self.root/'run'
        with patch('alphase.engine.evaluate',return_value={'image_macro_miou':.5,'class_macro_iou':[.5]*4}):train('synthetic',cfg,out)
        (out/'best.pt').write_bytes(b'corrupted derived checkpoint')
        train('synthetic',cfg,out,resume=out/'last.pt')
        best=torch.load(out/'best.pt',weights_only=False);last=torch.load(out/'last.pt',weights_only=False)
        torch.testing.assert_close(best['model'],last['best_eval_model'],rtol=0,atol=0)

    def test_early_stopped_run_has_zero_remaining_cost(self):
        cfg=self.training_context();out=self.root/'run'
        with patch('alphase.engine.evaluate',return_value={'image_macro_miou':.5,'class_macro_iou':[.5]*4}):train('synthetic',cfg,out)
        cost=estimate(out,.5,self.root/'cost.json')
        self.assertTrue(cost['completed']);self.assertEqual(cost['measured_epochs'],2)
        self.assertEqual(cost['estimated_remaining_compute_usd'],0)
        for rate in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):estimate(out,rate,self.root/'invalid.json')

    def test_evaluation_directory_cannot_mix_old_caches(self):
        save_json(self.root/'evaluation/stale.npz',{'previous':'model'})
        with self.assertRaisesRegex(ValueError,'empty evaluation'):evaluate(None,'not_read',self.root/'evaluation',cache=False)

    def test_corrupted_cache_or_metadata_is_rejected(self):
        y=np.array([[0,1],[2,3]],np.uint8);p=np.eye(4,dtype=np.float32)[y].transpose(2,0,1)
        record=write_cache(self.root/'cache',p,y,DEFAULT)
        search([self.root/'cache'],self.root/'ok.json')
        record['checkpoint_sha256']=['different'];save_json(self.root/'cache/metrics.json',record)
        with self.assertRaisesRegex(ValueError,'provenance'):search([self.root/'cache'],self.root/'bad.json')
        record['prediction_fingerprint']=prediction_fingerprint(record);save_json(self.root/'cache/metrics.json',record)
        with self.assertRaisesRegex(ValueError,'provenance'):search([self.root/'cache'],self.root/'bad.json')

    def prepare_comparison(self,ids,seeds=(42,)):
        plan(DEFAULT,self.root/'plan',seeds,ids)
        return { (ident,seed):load_config(self.root/'plan'/f'{ident}_s{seed}.json') for ident in ids for seed in seeds}

    def write_metric(self,ident,seed,record,tta=False):
        save_json(self.root/'runs'/f'{ident}_s{seed}'/('evaluation_tta' if tta else 'evaluation')/'metrics.json',record)

    def comparison(self):return compare(self.root/'plan/plan.json',self.root/'runs',self.root/'comparison')

    def test_tta_requires_same_checkpoint_and_actual_tta(self):
        c=self.prepare_comparison(['E'])['E',42];self.write_metric('E',42,metric_record(c,.5))
        for tta,checkpoint in [('none','synthetic_checkpoint'),('d4','unrelated')]:
            self.write_metric('E',42,metric_record(c,.7,tta=tta,checkpoint=checkpoint),True)
            with self.assertRaisesRegex(ValueError,'TTA'):self.comparison()
        self.write_metric('E',42,metric_record(c,.7,tta='d4'),True)
        self.assertEqual(self.comparison()['suggested_variant'],'E_TTA')

    def test_unmatched_budgets_and_smoke_cannot_win(self):
        configs=self.prepare_comparison(['E','F'])
        configs['F',42]['epochs']=80;save_json(self.root/'plan/F_s42.json',configs['F',42])
        self.write_metric('E',42,metric_record(configs['E',42],.5));self.write_metric('F',42,metric_record(configs['F',42],.8))
        r=self.comparison();self.assertEqual(r['suggested_variant'],'E');self.assertIn('unmatched_training_budget',r['exclusions']['F'])
        self.write_metric('E',42,metric_record(configs['E',42],.5,steps=1))
        self.assertIsNone(self.comparison()['suggested_variant'])

    def test_majority_of_seeds_required(self):
        configs=self.prepare_comparison(['E','F'],[42,123,2026])
        for seed,delta in [(42,-.005),(123,-.005),(2026,.1)]:
            self.write_metric('E',seed,metric_record(configs['E',seed],.5))
            self.write_metric('F',seed,metric_record(configs['F',seed],.5+delta))
        r=self.comparison();self.assertEqual(r['suggested_variant'],'E');self.assertEqual(r['selection_gates'][0]['positive_seeds'],1)

    def test_missing_parent_and_legacy_metrics_cannot_win(self):
        c=self.prepare_comparison(['F'])['F',42];self.write_metric('F',42,metric_record(c,.8))
        self.assertIsNone(self.comparison()['suggested_variant'])
        save_json(self.root/'runs/F_s42/evaluation/metrics.json',{'absent_policy':'exclude'})
        with self.assertRaisesRegex(ValueError,'legacy'):self.comparison()

    def test_missing_budget_control_cannot_yield_replicated_winner(self):
        configs=self.prepare_comparison(['E','CN'],[42,123,2026])
        for seed in [42,123,2026]:self.write_metric('CN',seed,metric_record(configs['CN',seed],.8))
        self.write_metric('E',42,metric_record(configs['E',42],.5))
        r=self.comparison();self.assertIsNone(r['suggested_variant'])
        self.assertEqual(r['selection_status'],'no_eligible_candidate')
        self.assertIn('missing_budget_control',r['exclusions']['CN'])

    def test_steps_zero_never_dispatches_training(self):
        with patch('alphase.experiments.subprocess.run') as dispatch:
            with self.assertRaises(ValueError):run('not_read','not_read',self.root/'not_created',steps=0)
            dispatch.assert_not_called()

    def test_confirmation_strings_are_not_permission(self):
        save_json(self.root/'contract.json',{'output_format_confirmed':'false'})
        with self.assertRaises(ValueError):load_contract(self.root/'contract.json')


if __name__=='__main__':unittest.main()
