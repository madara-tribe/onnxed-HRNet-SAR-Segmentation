import numpy as np

def gauss_aug(img):
    row,col,ch= img.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return img + gauss
	
def salt(img, salt=True):
	row,col,ch = img.shape
	s_vs_p = 0.5
	amount = 0.004
	sp_img = img.copy()
	
	if salt:
		num_salt = np.ceil(amount * img.size * s_vs_p)
		coords = [np.random.randint(0, i-1 , int(num_salt)) for i in img.shape]
		sp_img[coords[:-1]] = (255,255,255)
	else:  # pepper
		num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in img.shape]
		sp_img[coords[:-1]] = (0,0,0)
	return sp_img
