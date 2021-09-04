import numpy as np

def calc_IoU(y, y_pred):
    by = np.array(y, dtype=bool)
    by_pred = np.array(y_pred, dtype=bool) 
    overlap = by * by_pred
    union = by + by_pred
    return overlap.sum() / float(union.sum())

