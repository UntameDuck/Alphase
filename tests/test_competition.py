"""v0.3 safety, small-data recipe, and inference regression contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from alphase.config import DEFAULT,load_config,validate
from alphase.models import build_model,ModelEMA,make_optimizer
from alphase.sampling import PatchDataset,phase_boundary
from alphase.losses import loss_value,classwise_boundary_loss
from alphase.inference import predict,transform,transforms,tta_mode
from alphase.engine import atomic_save,load_model
from alphase.experiments import plan
from alphase.data import PALETTE
from alphase.resources import benchmark


class Pointwise(torch.nn.Module):
    def forward(self,x):return torch.cat([x,x[:,:1]],1)


class CompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)

    def test_invalid_numeric_and_schedule(self):
        for delta in [{'lr':float('nan')},{'batch_size':1.5},{'pretrained':1},
                      {'warmup_epochs':60},{'ema_decay':1.},{'accumulation_steps':0},
                      {'class_weights':[1,1,float('inf'),1]}, {'phase_probability':.9,'boundary_probability':.2},
                      {'scale_range':[0,1]},{'normalization_std':[1,0,1]}]:
            with self.subTest(delta=delta),self.assertRaises(ValueError):validate(DEFAULT|delta)

    def test_pretrained_requires_permission_before_download(self):
        with patch('alphase.models.smp.UnetPlusPlus') as factory:
            with self.assertRaises(ValueError):build_model(DEFAULT|{'pretrained':True})
            factory.assert_not_called()

    def test_cpu_benchmark_is_not_cost_evidence(self):
        result=benchmark('not_read_on_cpu',DEFAULT|{'device':'cpu'})
        self.assertEqual(result['status'],'skipped_no_cuda')
        self.assertNotIn('median_batch_seconds',result)

    def test_challenger_forward_backward_and_normalization(self):
        cfg=load_config(Path(__file__).resolve().parents[1]/'configs'/'challenger.json')
        model=build_model(cfg,download=False)
        x=torch.randn(2,3,64,96);y=model(x)
        self.assertEqual(tuple(y.shape),(2,4,64,96))
        loss_value(y,torch.randint(0,4,(2,64,96)),cfg).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        import segmentation_models_pytorch as smp
        for encoder in [cfg['encoder'],'timm-efficientnet-b3']:
            # Force the bundled fallback rather than fetching preprocessing metadata.
            with patch('segmentation_models_pytorch.encoders.hf_hub_download',side_effect=OSError('offline test')):
                prep=smp.encoders.get_preprocessing_params(encoder)
            np.testing.assert_allclose(prep['mean'],DEFAULT['normalization_mean'])
            np.testing.assert_allclose(prep['std'],DEFAULT['normalization_std'])

    def test_d4_inverse_and_batched_seams(self):
        x=torch.arange(2*4*7*11).reshape(2,4,7,11)
        for k,flip in transforms('d4'):self.assertTrue(torch.equal(transform(transform(x,k,flip),k,flip,True),x))
        rng=np.random.default_rng(44);rgb=rng.integers(0,256,(93,117,3),dtype=np.uint8)
        a=predict(Pointwise(),rgb,torch.device('cpu'),64,16,False,1)
        for mode in ['flip','d4']:
            b=predict(Pointwise(),rgb,torch.device('cpu'),32,8,mode,3)
            np.testing.assert_allclose(a,b,atol=2e-6)
            np.testing.assert_allclose(b.sum(0),1,atol=1e-6)
        self.assertEqual(tta_mode(False),'none')
        with self.assertRaises(ValueError):tta_mode('arbitrary')

    def test_classwise_boundaries_keep_phase_identity(self):
        y=torch.zeros((1,16,16),dtype=torch.long);y[:,:,8:]=1
        gt=F.one_hot(y,4).permute(0,3,1,2).float();valid=torch.ones_like(y,dtype=torch.bool)
        correct=classwise_boundary_loss(gt,gt,valid)
        wrong=classwise_boundary_loss(gt.roll(2,1),gt,valid)
        self.assertLess(float(correct),1e-6);self.assertGreater(float(wrong),.9)

    def test_image_loss_padding_and_missing_classes(self):
        cfg=DEFAULT|{'dice_reduction':'image_present','boundary_weight':.05}
        y=torch.full((2,32,32),3,dtype=torch.long);y[0,:8]=0;y[:,30:]=255
        z=torch.randn(2,4,32,32,requires_grad=True)
        loss=loss_value(z,y,cfg);loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertEqual(float(z.grad[:,:,30:].abs().sum()),0.)
        modified=z.detach().clone();modified[:,:,30:]=100
        torch.testing.assert_close(loss.detach(),loss_value(modified,y,cfg))
        y[0]=255
        with self.assertRaises(ValueError):loss_value(z,y,cfg)

    def test_sampler_alignment_absent_fallback_and_epoch_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'train/images').mkdir(parents=True);(root/'train/masks').mkdir()
            y=np.full((96,96),3,np.uint8);y[:24]=2
            Image.fromarray(PALETTE[y]).save(root/'train/images/a_image.png')
            Image.fromarray(PALETTE[y]).save(root/'train/masks/a_mask.png')
            cfg=DEFAULT|{'patch_size':64,'overlap':16,'sampler':'phase_boundary','cache_images':True,
                'phase_probability':1.,'boundary_probability':0.,'brightness':False}
            ds=PatchDataset(root,cfg);x,t=ds[0];x2,t2=ds[0]
            self.assertTrue(torch.equal(x,x2) and torch.equal(t,t2));self.assertIn(2,t.unique())
            restored=((x.numpy().transpose(1,2,0)*np.array(cfg['normalization_std'])+cfg['normalization_mean'])*255).round().astype(np.uint8)
            np.testing.assert_array_equal(restored,PALETTE[t.numpy()])
            ds.set_epoch(1);a,b=ds[0];ds.set_epoch(0);x3,t3=ds[0]
            self.assertTrue(torch.equal(x,x3) and torch.equal(t,t3));self.assertFalse(torch.equal(t,b))
            np.testing.assert_array_equal(ds.cache[0][1],y)
            # Neither requested phase exists: uniform fallback must still work.
            ds.cfg['phase_classes']=[0];ds.cache.clear();self.assertEqual(tuple(ds[0][1].shape),(64,64))
            self.assertFalse(phase_boundary(np.full((4,4),255,np.uint8)).any())

    def test_ema_export_and_optimizer_groups(self):
        cfg=DEFAULT|{'architecture':'unet','encoder':'resnet18','ema_decay':.99,'encoder_lr':1e-4}
        model=build_model(cfg,False);ema=ModelEMA(model,.99);before=next(ema.module.parameters()).detach().clone()
        with torch.no_grad():next(model.parameters()).add_(1.)
        ema.update(model);after=next(ema.module.parameters())
        self.assertTrue(torch.all(after>before));self.assertTrue(torch.all(after<before+1.))
        opt=make_optimizer(model,cfg);self.assertEqual([g['lr'] for g in opt.param_groups],[1e-4,3e-4])
        ids=[id(p) for g in opt.param_groups for p in g['params']];self.assertEqual(len(ids),len(set(ids)))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'best.pt';atomic_save(path,{'config':cfg,'model':model.state_dict(),'eval_model':ema.module.state_dict()})
            loaded,_=load_model(path,torch.device('cpu'))
            for key,value in loaded.state_dict().items():torch.testing.assert_close(value,ema.module.state_dict()[key])

    def test_experiments_reset_contaminated_base(self):
        with tempfile.TemporaryDirectory() as d:
            base=DEFAULT|{'architecture':'deeplabv3plus','encoder':'resnet50','boundary_weight':.9,'sampler':'phase_boundary','rare_probability':.7,'loss_kind':'ce_tversky'}
            plan(base,d,[42],['A','D','E','F','G','CN'])
            configs={ident:load_config(Path(d)/(ident+'_s42.json')) for ident in ['A','D','E','F','G','CN']}
            for ident in ['A','D','E']:self.assertEqual(configs[ident]['boundary_weight'],0);self.assertEqual(configs[ident]['sampler'],'uniform')
            self.assertEqual(configs['D']['architecture'],'unetplusplus');self.assertEqual(configs['D']['encoder'],'timm-efficientnet-b3')
            self.assertEqual(configs['F']['sampler'],'phase_boundary');self.assertEqual(configs['G']['boundary_weight'],.05)
            self.assertFalse(json.loads((Path(d)/'plan.json').read_text(encoding='utf-8'))['run_ensemble'])
            with self.assertRaises(ValueError):plan(base,d,[42],['A'])


if __name__=='__main__':unittest.main()
