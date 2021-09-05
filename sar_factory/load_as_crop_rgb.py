import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from equalizeHist import equalizeHist


class TIFLoad():
    def __init__(self):
        self.resize_w, self.resize_h = 448, 448
        self.size = 448*2
        self.input_chanel = 3
        self.num_folder = 28
        
    def quarter_crop(self, img, size=448):
        c1 = img[0:size, 0:size]
        c2 = img[0:size, size:size*2]
        c3 = img[size:size*2, 0:size]
        c4 = img[size:size*2, size:size*2]
        return [c1, c2, c3, c4]
  
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
        ims = (im3-im1)+(im4-im2)
        bgr = self.create_bgr(ims, im1, im2)
        bgr = self.clipping(bgr, clip_max=255)
        bgr = equalizeHist(bgr)
        #c, h, w = im_out.shape
        images = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        images = self.tif_normalize(images, StandardScaler=None)
        
        anno = cv2.imread('./train_annotations/train_{}.png'.format(idx), -1)
        anno = cv2.resize(anno, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        print(np.array(anno).shape, np.array(images).shape)
        return images, anno

    def run(self):
        imgs, annos = [], []
        for idx in range(self.num_folder):
            if idx<10:
                idx = '0'+str(idx)        
            images, anno = self.tif_load(idx)
            im_crops = self.quarter_crop(images, size=self.resize_w)
            anno_crops = self.quarter_crop(anno, size=self.resize_w)
            #plt.imshow(images),plt.show()
            #plt.imshow(anno),plt.show()
            for i in range(4):
                imgs.append(im_crops[i])
                annos.append(anno_crops[i])
            
        imgs_ = np.array(imgs).reshape(len(imgs), self.resize_w, self.resize_h, self.input_chanel)
        annos_ = np.array(annos).reshape(len(annos), self.resize_w, self.resize_h, 1)
        print(imgs_.shape, annos_.shape)
        return imgs_, annos_

    
if __name__=='__main__':
    imgs, annos = TIFLoad().run()
    print(imgs.max(), imgs.min(), np.unique(annos), imgs.shape, annos.shape)
    np.save('data/sar_img', imgs)
    np.save('data/sar_anno', annos)


