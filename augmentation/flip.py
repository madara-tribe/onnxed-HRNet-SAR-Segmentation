import numpy as np

def npflip(img, types='up'):
    if types=='up':
        img = img[::-1]
    elif types=='lr':
        img = img[:, ::-1]
    elif types=='lrup':
        img = img[::-1, ::-1]
    return img

