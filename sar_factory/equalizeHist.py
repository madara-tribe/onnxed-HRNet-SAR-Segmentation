import cv2

def equalizeHist(img):
    for j in range(3):
        img[:, :, j] = cv2.equalizeHist(img[:, :, j])
    return img
