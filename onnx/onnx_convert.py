import os, sys
sys.path.append('../')
os.environ['TF_KERAS'] = '1'
import numpy as np
import keras2onnx
from keras2onnx import convert_keras
import onnxruntime
import onnx
from option_parser import get_option
from model.seg_hrnet import seg_hrnet


weight_dir = '../weights'
weight_name = 'cp-08.hdf5'
OUTPUT_ONNX_MODEL_NAME = 'hrnet.onnx'
H=448
W=448
C=3


def load_model(config=None, weight_path=None):
    models = seg_hrnet(H, W, C, config.num_cls)
    if weight_path:
        models.load_weights(os.path.join(weight_dir, weight_path))
    #models.summary()
    return models

def main(config):
    onnx_model_file_name = OUTPUT_ONNX_MODEL_NAME
    models = load_model(config=config, weight_path=weight_name)
    print(models.name)
    
    onnx_model = convert_keras(models, models.name)
    onnx.save(onnx_model, onnx_model_file_name)
    print("success to output " + onnx_model_file_name)

if __name__=='__main__':
    cfg = get_option()
    main(cfg)

