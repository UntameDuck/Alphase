import tempfile
import unittest
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from alphase.config import DEFAULT,resolved_config
from alphase.losses import lovasz_image_loss,loss_value
from alphase.models import build_model
from alphase.sampling import component_coordinates,PatchDataset
from alphase.engine import atomic_save,load_model
from alphase.data import PALETTE
from alphase.morphology import audit_train
from alphase.metrics import diagnostics,summarize
from PIL import Image


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)

    def test_lovasz_matches_reference_values_and_gradients(self):
        torch.manual_seed(73)
        a=torch.randn(2,4,9,11,requires_grad=True);b=a.detach().clone().requires_grad_()
        y=torch.randint(0,3,(2,9,11));y[0]=3;y[:,0]=255
        loss=lovasz_image_loss(a.softmax(1),y)
        reference=smp.losses.LovaszLoss(mode='multiclass',per_image=True,ignore_index=255)(b,y)
        torch.testing.assert_close(loss,reference);loss.backward();reference.backward()
        torch.testing.assert_close(a.grad,b.grad);self.assertEqual(float(a.grad[:,:,0].abs().sum()),0.)

    def test_lovasz_perfect_and_equal_image_weight(self):
        y=torch.tensor([[[0,1],[2,3]],[[3,3],[3,255]]]);p=F.one_hot(y.clamp_max(3),4).permute(0,3,1,2).float()
        self.assertEqual(float(lovasz_image_loss(p,y)),0.)
        q=.7*p+.3/4
        torch.testing.assert_close(lovasz_image_loss(q,y),(lovasz_image_loss(q[:1],y[:1])+lovasz_image_loss(q[1:],y[1:]))/2)
        with self.assertRaises(ValueError):lovasz_image_loss(p,torch.full_like(y,255))

    def test_scse_combined_candidate_forward_backward_reload(self):
        cfg=resolved_config(DEFAULT|{'decoder_attention':'scse','lovasz_weight':.2,'dice_reduction':'image_present','boundary_weight':.05})
        model=build_model(cfg,False);x=torch.randn(2,3,64,64);out=model(x)
        self.assertEqual(tuple(out.shape),(2,4,64,64));loss_value(out,torch.randint(0,4,(2,64,64)),cfg).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(any('attention' in name and 'cSE' in name for name,_ in model.named_parameters()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';atomic_save(path,{'model':model.state_dict(),'config':cfg})
            loaded,_=load_model(path,torch.device('cpu'));model.eval()
            with torch.inference_mode():torch.testing.assert_close(model(x),loaded(x))

    def test_component_groups_balance_regions_not_pixels(self):
        y=np.full((32,32),3,np.uint8);y[1:3,1:3]=2;y[10:20,10:20]=2;y[30,30]=2
        groups=component_coordinates(y,2,4)
        self.assertEqual(sorted(map(len,groups)),[4,100]);self.assertTrue(all((y.ravel()[g]==2).all() for g in groups))
        rng=random.Random(7);small=sum(len(rng.choice(groups))==4 for _ in range(1000))
        self.assertTrue(400<small<600)
        self.assertEqual(component_coordinates(y,0,4),[])

    def test_component_sampler_morphology_and_mask_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'train/images').mkdir(parents=True);(root/'train/masks').mkdir()
            y=np.full((96,96),3,np.uint8);y[8:11,8:11]=2;y[30:70,30:70]=0
            Image.fromarray(PALETTE[y]).save(root/'train/images/a_image.png');Image.fromarray(PALETTE[y]).save(root/'train/masks/a_mask.png')
            cfg=DEFAULT|{'patch_size':64,'overlap':16,'sampler':'phase_boundary','phase_sampling':'component','phase_probability':1.,'boundary_probability':0.,'phase_classes':[2],'cache_images':True,'brightness':False}
            ds=PatchDataset(root,cfg);x,t=ds[0];x2,t2=ds[0]
            self.assertTrue(torch.equal(x,x2) and torch.equal(t,t2));self.assertIn(2,t.unique())
            restored=((x.numpy().transpose(1,2,0)*cfg['normalization_std']+cfg['normalization_mean'])*255).round().astype(np.uint8)
            np.testing.assert_array_equal(restored,PALETTE[t.numpy()])
            r=audit_train(root);self.assertEqual(r['image_count'],1);self.assertEqual(r['classes'][2]['area_quantiles']['p50'],9)

    def test_area_fraction_error_is_image_macro(self):
        y=np.array([[0,0],[1,1]],np.uint8);p=np.array([[0,1],[1,1]],np.uint8)
        r=diagnostics(p,y);summary=summarize([r])
        self.assertEqual(summary['class_macro_area_fraction_mae'],[.25,.25,0.,0.])

    def test_new_config_invalid_combinations(self):
        for values in [{'decoder_attention':'unknown'},{'architecture':'deeplabv3plus','decoder_attention':'scse'},
                       {'lovasz_weight':-1},{'phase_sampling':'unknown'},{'component_min_area':0},{'seed':-1},{'min_epochs':61}]:
            with self.subTest(values=values),self.assertRaises(ValueError):resolved_config(values)


if __name__=='__main__':unittest.main()
