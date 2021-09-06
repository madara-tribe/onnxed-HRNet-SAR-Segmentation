import cv2

def blur(img):
    average_square = (10,10)
    return cv2.blur(img, average_square)
