import numpy as np

def create_binary_mask(pred, threshold=0.1):
    markers = np.zeros_like(pred)
    markers[pred < threshold] = 0
    markers[pred > threshold] = 1
    #print(np.unique(markers))
    return markers
