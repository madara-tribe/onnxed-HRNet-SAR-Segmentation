import sys
sys.path.append('../')
import cv2
import numpy as np
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils.cal_iou import calc_IoU
from utils.TTA import run_tta
from utils.binary_mask import create_binary_mask
from sar_factory.equalizeHist import equalizeHist



class TIFeval():
    def __init__(self, tta=True, mask_threshold=0.9):
        self.resize_w, self.resize_h = 448, 448
        self.size = 448*2
        self.input_chanel = 3
        self.num_folder = 28
        self.tta = tta
        self.mask_threshold = mask_threshold
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
        im1 = cv2.imread('./train_images/train_{}/0_VV.tif'.format(idx), -1)
        im2 = cv2.imread('./train_images/train_{}/0_VH.tif'.format(idx), -1)
        im3 = cv2.imread('./train_images/train_{}/1_VV.tif'.format(idx), -1)
        im4 = cv2.imread('./train_images/train_{}/1_VH.tif'.format(idx), -1)
        ims = (im3 - im1)+(im4-im2)
        bgr = self.create_bgr(ims, im1, im2)
        bgr = self.clipping(bgr, clip_max=255)
        bgr = equalizeHist(bgr)
        
        images = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        images = self.tif_normalize(images, StandardScaler=None)
        
        anno = cv2.imread('./train_annotations/train_{}.png'.format(idx), -1)
        anno = cv2.resize(anno, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return images, anno

    def predict(self, model):
        preds, annos = [], []
        IOU=0
        for idx in range(self.num_folder):
            if idx<10:
                idx = '0'+str(idx)        
            images, anno = self.tif_load(idx)
            #print(images.shape, anno.shape)
            pred_img = self.predict_quarter_crop(model, images, size=self.resize_w, c=self.input_chanel)
            preds.append(pred_img)
            annos.append(anno)
            # IOU with create_binary_mask
            b_mask = create_binary_mask(pred_img, threshold=self.mask_threshold)
            print(calc_IoU(anno, b_mask))
            IOU += calc_IoU(anno, b_mask)
        print('total IOU', IOU/len(preds))
        return np.array(anno), np.array(preds)
    
if __name__=='__main__':
    anno, preds = TIFeval().predict(model)
    print(preds.shape, annos.shape, preds.max(), preds.min(), np.unique(annos))
    np.save('saved_preds', preds)



