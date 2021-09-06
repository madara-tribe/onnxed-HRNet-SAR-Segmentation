import numpy as np
def gause_noise(img):
    row,col,ch= img.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = img + gauss
    return gauss_img
