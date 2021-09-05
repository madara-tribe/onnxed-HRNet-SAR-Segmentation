import numpy as np
import cv2

def contrast(img, gamma = 0.5):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')

    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    return cv2.LUT(img, gamma_cvt)
