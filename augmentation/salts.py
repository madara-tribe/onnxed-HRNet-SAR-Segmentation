import numpy as np

def salt(img):
    row,col,ch = img.shape
    s_vs_p = 0.5
    amount = 0.004
    sp_img = img.copy()
    # pepper
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in img.shape]
    sp_img[coords[:-1]] = (0,0,0)
    return sp_img
