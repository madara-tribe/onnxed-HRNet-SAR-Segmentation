import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from equalizeHist import equalizeHist

class TIFLoad():
    def __init__(self):
        self.resize_w, self.resize_h = 448, 448
        self.input_chanel = 3
        self.num_folder = 28

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
        #c, h, w = im_out.shape
        images = cv2.resize(bgr, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        anno = cv2.imread('./train_annotations/train_{}.png'.format(idx), -1)
        anno = cv2.resize(anno, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
    
        images = self.tif_normalize(images, StandardScaler=None)
        print(np.array(anno).shape, np.array(images).shape)
        return images, anno

    def run(self):
        imgs, annos = [], []
        for idx in range(self.num_folder):
            if idx<10:
                idx = '0'+str(idx)        
            images, anno = self.tif_load(idx)
            plt.imshow(images),plt.show()
            plt.imshow(anno),plt.show()
            print(anno.shape, images.shape, images.min(), images.max(), anno.min(), anno.max())
            imgs.append(images)
            annos.append(anno)
        imgs = np.array(imgs).reshape(self.num_folder, self.resize_w, self.resize_h, self.input_chanel)
        annos = np.array(annos).reshape(self.num_folder, self.resize_w, self.resize_h, 1)
        print(imgs.shape, annos.shape)
        return imgs, annos

if __name__=='__main__':
    imgs, annos = TIFLoad().run()
    print(imgs.max(), imgs.min(), np.unique(annos), imgs.shape, annos.shape)
    np.save('data/sar_img', imgs)
    np.save('data/sar_anno', annos)



