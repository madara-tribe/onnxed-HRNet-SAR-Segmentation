import sys, os, cv2
sys.path.append('../')
import onnxruntime
import onnx
import numpy as np
from utils.cal_iou import calc_IoU 
from utils.fbeta_score import eager_binary_fbeta
from utils.binary_mask import create_binary_mask 
H = W = 448

def load_data(config=None):
    imgs, annos = np.load('../data/sar_img.npy'), np.load('../data/sar_anno.npy')
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    print(imgs.shape, annos.shape)
    return imgs, annos

def onnx_inference(onnx_path):
    imgs, annos = load_data()
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    IOU = 0
    for i, im in enumerate(imgs):
        pred_annos = session.run(None, {input_name: im.reshape(1, H, W, 3)})[0]
        pred_annos = create_binary_mask(pred_annos, threshold=0.9)
        print('iou score', calc_IoU(annos[i], pred_annos))
        IOU += calc_IoU(annos[i], pred_annos)
    print('total IOU score', IOU/len(annos))
if __name__=='__main__':
    onnx_path = str(sys.argv[1])
    onnx_inference(onnx_path)
