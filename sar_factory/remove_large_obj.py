import numpy as np
import cv2

def remove(im2, im1):
    im_out = np.maximum(im2 - im1, 0.5) - 0.5
    im_out = np.heaviside(im_out, 0)
    im_out = remove_blob(im_out * 255, threshold_blob_area=25)
    return im_out

def remove_blob(im_in, threshold_blob_area=25):
    '''remove small blob from your image '''
    im_out = im_in.copy()
    contours, hierarchy = cv2.findContours(im_in.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range (1, len(contours)):
        index_level = int(hierarchy[0][i][1])
        if index_level <= i:
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area <= threshold_blob_area:
                cv2.drawContours(im_out, [cnt], -1, 0, -1, 1)
    return im_out
