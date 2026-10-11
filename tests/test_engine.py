import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from alphase.config import DEFAULT,validate
from alphase.engine import predict,loss_value,build_model,load_model,atomic_save,normalized_weights
from alphase.resources import tune

class Pointwise(torch.nn.Module):
    def forward(self,x):return torch.cat([x,x[:,:1]],1)

class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
    def test_tiling_tta(self):
        model=Pointwise();rng=np.random.default_rng(2)
        for shape in [(13,17),(65,93),(64,64)]:
            x=rng.integers(0,255,(*shape,3),dtype=np.uint8);a=predict(model,x,torch.device('cpu'),64,16);b=predict(model,x,torch.device('cpu'),32,8,True)
            self.assertEqual(a.shape,(4,*shape));np.testing.assert_allclose(a,b,atol=1e-6);np.testing.assert_allclose(a.sum(0),1,atol=1e-6)
    def test_losses_ignore_gradient(self):
        for kind in ['ce_dice','focal_dice','ce_tversky']:
            cfg=DEFAULT|{'loss_kind':kind,'boundary_weight':.05};y=torch.randint(0,4,(2,32,32));y[:,:2]=255
            x=torch.randn(2,4,32,32,requires_grad=True);loss=loss_value(x,y,cfg);loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all());modified=x.detach().clone();modified[:,:,:2]=100
            self.assertAlmostEqual(float(loss.detach()),float(loss_value(modified,y,cfg)),places=5)
    def test_core_reload(self):
        cfg=DEFAULT.copy();model=build_model(cfg);x=torch.randn(2,3,64,64);out=model(x);self.assertEqual(out.shape,(2,4,64,64));out.square().mean().backward();model.eval()
        with torch.inference_mode():before=model(x)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.pt';atomic_save(path,{'config':cfg,'model':model.state_dict()});m,_=load_model(path,torch.device('cpu'))
            with torch.inference_mode():after=m(x)
            torch.testing.assert_close(before,after)
    def test_cpu_tune_is_explicit_skip(self):
        cfg=DEFAULT|{'device':'cpu'};updated,record=tune(cfg);self.assertEqual(updated,cfg);self.assertEqual(record['status'],'skipped_no_cuda')
    def test_weight_validation(self):
        np.testing.assert_allclose(normalized_weights([1,3],2),[.25,.75])
        for weights in [[0,0],[-1,2],[float('nan'),1],[1]]:
            with self.assertRaises(ValueError):normalized_weights(weights,2)
    def test_remaining_model_factories(self):
        for arch,encoder in [('unet','resnet34'),('unet','timm-efficientnet-b3'),('unetplusplus','resnet34'),('deeplabv3plus','resnet50')]:
            cfg=DEFAULT|{'architecture':arch,'encoder':encoder};m=build_model(cfg)
            out=m(torch.randn(2,3,64,64));self.assertEqual(tuple(out.shape),(2,4,64,64));out.square().mean().backward()
    def test_bad_config(self):
        with self.assertRaises(ValueError):validate(DEFAULT|{'patch_size':63})

if __name__=='__main__':unittest.main()
