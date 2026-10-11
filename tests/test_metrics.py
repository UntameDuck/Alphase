import unittest
import numpy as np
from alphase.data import PALETTE,decode_mask
from alphase.metrics import image_iou,boundary_f1,small_components,paired_bootstrap
from alphase.postprocess import clean

class MetricsTests(unittest.TestCase):
    def test_palette(self):
        x=np.array([[0,1],[2,3]],np.uint8);np.testing.assert_array_equal(decode_mask(PALETTE[x]),x)
        with self.assertRaises(ValueError):decode_mask(np.zeros((1,1,3),np.uint8))
    def test_manual_iou(self):
        y=np.array([[0,0],[1,1]]);p=np.array([[0,1],[1,1]])
        self.assertAlmostEqual(image_iou(p,y)[0],(.5+2/3)/2)
        self.assertAlmostEqual(image_iou(p,y,'zero')[0],(.5+2/3)/4)
        self.assertAlmostEqual(image_iou(p,y,'one')[0],(.5+2/3+2)/4)
    def test_false_positive_absent(self):
        self.assertEqual(image_iou(np.ones((3,3),int),np.zeros((3,3),int))[0],0)
    def test_macro_not_global(self):
        a=image_iou(np.zeros((10,10),int),np.zeros((10,10),int))[0]
        b=image_iou(np.ones((1,1),int),np.zeros((1,1),int))[0];self.assertEqual((a+b)/2,.5)
    def test_boundary_tolerance(self):
        a=np.zeros((20,20),bool);a[4:12,4:12]=True;b=np.roll(a,1,axis=1)
        self.assertEqual(boundary_f1(a,a,0),1)
        self.assertEqual(boundary_f1(a,b,1),1)
        self.assertLess(boundary_f1(a,b,0),1)
        self.assertIsNone(boundary_f1(np.zeros_like(a),np.zeros_like(a)))
    def test_small_recall_not_false_iou(self):
        y=np.zeros((10,10),bool);y[1,1]=True;y[5:7,5:7]=True
        p=np.zeros_like(y);p[1,1]=True;p[9,9]=True
        r=small_components(p,y,4);self.assertEqual(r['gt_small_count'],2)
        self.assertEqual(r['gt_small_pixel_recall'],.2);self.assertEqual(r['unmatched_small_prediction_count'],1)
    def test_paired_bootstrap(self):
        a=[{'image':'a','miou':.4},{'image':'b','miou':.5}];b=[{'image':'a','miou':.5},{'image':'b','miou':.6}]
        r=paired_bootstrap(a,b);self.assertAlmostEqual(r['mean_delta'],.1);self.assertEqual(r['improved_images'],2)
        with self.assertRaises(ValueError):paired_bootstrap(a,b[:1])
    def test_confidence_clean(self):
        p=np.zeros((4,5,5),np.float32);p[3]=1;p[:,2,2]=[.4,.1,.1,.4];p[0,2,2]=.41;p[3,2,2]=.39
        self.assertEqual(clean(p,2,.6)[2,2],3)
        p[:,2,2]=[.9,0,0,.1];self.assertEqual(clean(p,2,.6)[2,2],0)

if __name__=='__main__':unittest.main()
