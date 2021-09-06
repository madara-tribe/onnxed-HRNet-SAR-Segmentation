import sys
sys.path.append('../')
import cv2
import numpy as np
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils.TTA import run_tta
from sar_factory.equalizeHist import equalizeHist
from option_parser import get_option
from model.seg_hrnet import seg_hrnet

H=448
W=448
C=3

def load_model(config=None, weight_path=None):
    models = seg_hrnet(H, W, C, config.num_cls)
    models.load_weights(os.path.join('weights', weight_path))
    models.summary()
    return models
    
    
class TEST_pred():
    def __init__(self, tta=True):
        self.resize_w, self.resize_h = 448, 448
        self.size = 448*2
        self.input_chanel = 3
        self.num_folder = 21
        self.tta = tta
        
    def predict_quarter_crop(self, m, img, size=448, c=3):
        c1 = img[0:size, 0:size].reshape(1, size, size, c)
        c2 = img[0:size, size:size*2].reshape(1, size, size, c)
        c3 = img[size:size*2, 0:size].reshape(1, size, size, c)
        c4 = img[size:size*2, size:size*2].reshape(1, size, size, c)
        if self.tta is True:
            print('TTA pred')
            p1, p2 = run_tta(m, c1), run_tta(m, c2)
            p3, p4 = run_tta(m, c3), run_tta(m, c4)
        else:
            print('No TTA pred')
            p1, p2 = m.predict(c1), m.predict(c2)
            p3, p4 = m.predict(c3), m.predict(c4)
        pred_image = self.quarter2all(p1, p2, p3, p4, size=size)
        return pred_image
   
    def quarter2all(self, c1, c2, c3, c4, size=448):
        block1 = np.hstack([c1.reshape(size, size), c2.reshape(size, size)])
        block2 = np.hstack([c3.reshape(size, size), c4.reshape(size, size)])
        return np.vstack([block1, block2])
    
    def tif_normalize(self, img, StandardScaler=True):
        if StandardScaler:
            ss = preprocessing.StandardScaler()
            if img.ndim==4:
                b, w, h, c = img.shape
                img = img.reshape(b, w*h*c)
                img = ss.fit_transform(img)
                img = img.reshape(b, w, h, c)
            elif img.ndim==3:
                w, h, c = img.shape
                img = img.reshape(c, w*h)
                img = ss.fit_transform(img)
                img = img.reshape(w, h, c)
            elif img.ndim==2:
                img = ss.fit_transform(img)
        else:
            img = img/255
        return img

    def clipping(self, img, clip_max=99.5):
        #img = img_ * 255
        return img.clip(0, clip_max).astype('uint8')

    def create_bgr(self, img1, img2, img3=None):
        r = img1 * 255 
        g = img2 * 255 
        b = img3 * 255
        return np.dstack((r, g, b))
        
    def tif_load(self, idx):
        im1 = cv2.imread('./test_images/test_{}/0_VV.tif'.format(idx), -1)
        im2 = cv2.imread('./test_images/test_{}/0_VH.tif'.format(idx), -1)
        im3 = cv2.imread('./test_images/test_{}/1_VV.tif'.format(idx), -1)
        im4 = cv2.imread('./test_images/test_{}/1_VH.tif'.format(idx), -1)
        ims = (im3 - im1)+(im4-im2)
        bgr = self.create_bgr(ims, im1, im2)
        bgr = self.clipping(bgr, clip_max=255)
        bgr = equalizeHist(bgr)
        images = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        images = self.tif_normalize(images, StandardScaler=None)
        
        origin_img = cv2.imread('./test_images/test_{}/1_VH.tif'.format(idx), -1)
        return images, origin_img

    def predict(self, model):
        preds = []
        IOU=0
        for idx in range(self.num_folder):
            if idx<10:
                idx = '0'+str(idx)        
            images, ori_img = self.tif_load(idx)
            h, w = ori_img.shape
            pred_img = self.predict_quarter_crop(model, images, size=self.resize_w, c=self.input_chanel)
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_NEAREST)
            print(idx, pred_img.shape, ori_img.shape)
            preds.append(pred_img)
            # IOU with create_binary_mask
        return preds
    
if __name__=='__main__':
    cfg = get_option()
    model = load_model(config=cfg, weight_path='pre_ep10.hdf5')
    preds = TEST_pred(tta=None).predict(model)
    np.save('submit_preds', preds)




