import argparse
from core import calculate
import cv2, os
import numpy as np
import glob
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true-path', default = './test_annotations')
    parser.add_argument('--pred-path', default = './masks')

    args = parser.parse_args()

    return args

def cal_metrics(y_true, y_pred):
    if y_pred.max()==255:
        y_pred = (y_pred/255).astype(np.uint8)
    #print(np.unique(y_true), np.unique(y_pred))
    CM = confusion_matrix(y_true.flatten(), y_pred.flatten())
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print('FP, FN, TP, TN', FP, FN, TP, TN)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('accuracy', ACC)
    error, precision, recall = adapted_rand_error(y_true.flatten(), y_pred.flatten())
    print('precision, recall', precision, recall)
    return FP, FN, TP, TN, precision, recall, ACC


def main(args, range_threshold=0.1):
    c = 0
    ious = []
    fp, fn, tp, tn, precisions, recalls, accs = 0, 0, 0, 0, 0, 0, 0
    labels = [p for p in glob.glob(args.pred_path + '/*')]
    labels.sort()

    gt = [p for p in glob.glob(args.true_path + '/*')]
    gt.sort()
    N = len(gt)
    for y_true, y_pred in zip(gt, labels):
        y_true = cv2.imread(y_true, 0)
        y_pred = cv2.imread(y_pred, 0)
        if y_pred.shape!=y_true.shape:
            h, w = np.shape(y_pred)
            y_true = cv2.resize(y_true, (w, h))
        ious.append(calculate(y_true, y_pred, strict=True, iou_threshold=range_threshold))
        FP, FN, TP, TN, precision, recall, acc = cal_metrics(y_true, y_pred)
        fp +=FP
        fn += FN
        tp += TP
        tn += TN
        precisions += precision
        recalls += recall
        accs += acc
        c+=1
    print('FP, FN, TP, TN, precision, recall, ACC', int(fp/N), int(fn/N), int(tp/N), int(tn/N), precisions/N, recalls/N, accs/N)
    for r in ious:
        print(r.results)

if __name__=='__main__':
    args = parse_args()
    main(args)
