import numpy as np
import os, cv2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from model.seg_hrnet import seg_hrnet
from option_parser import get_option
from sar_factory.predict_as_crop import TIFeval

H=int(448)
W=int(448)
C=3


def load_model(config=None, weight_path=None):
    models = seg_hrnet(H, W, C, config.num_cls)
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.summary()
    return models


if __name__=='__main__':
    config=get_option()
    model = load_model(config, weight_path='cp-02.hdf5')
    annos, preds = TIFeval(tta=None).predict(model)
    print(preds.shape, annos.shape, preds.max(), preds.min(), np.unique(annos))
    np.save('preds', preds)




