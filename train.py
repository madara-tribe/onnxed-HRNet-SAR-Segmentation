import numpy as np
import os, cv2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from losses.bce_dice_loss import bce_dice_loss, dice_loss 
from utils.fbeta_score import binary_fbeta
from augmentation import flip, gause_noise, salts, Smoothing
from option_parser import get_option
from model.seg_hrnet import seg_hrnet

H=448
W=448
C=3

def noise_aug(X, y):
    X1 = [salts.salt(x_) for x_ in X]
    #X2 = [gause_noise.gause_noise(x_) for x_ in X]
    #X3 = [Smoothing.blur(x_) for x_ in X]
    X, y = np.vstack([X, X1]), np.vstack([y, y])
    return X, y

def flipaug(X, y):
    X1, y1 = flip.npflip(X, types='lr'), flip.npflip(y, types='lr')
    X2, y2 = flip.npflip(X, types='up'), flip.npflip(y, types='up')
    X3, y3 = flip.npflip(X, types='lrup'), flip.npflip(y, types='lrup')
    X, y = np.vstack([X, X1, X2, X3]), np.vstack([y, y1, y2, y3])
    return X, y

def load_data(config=None):
    imgs, annos = np.load('data/sar_img.npy'), np.load('data/sar_anno.npy')
    N = 10
    imgs, annos = imgs.astype(np.float32), annos.astype(np.float32)
    X_train, y_train = imgs[N:], annos[N:]
    X_val, y_val = imgs[:N], annos[:N]
    X_train, y_train = flipaug(X_train, y_train)
    #X_train, y_train = noise_aug(X_train, y_train)
    
    X_val, y_val = flipaug(X_val, y_val)
    print(X_train.shape, y_train.shape)
    return X_train, y_train, X_val, y_val

def load_model(config=None, weight_path=None):
    sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    keras_metrics = [binary_fbeta]
    models = seg_hrnet(H, W, C, config.num_cls)
    models.compile(optimizer=sgd, loss=bce_dice_loss, metrics=keras_metrics)
    if weight_path:
        models.load_weights(os.path.join('weights', weight_path))
    models.summary()
    return models


def call_callbacks():
    checkpoint_path = "weights/cp-{epoch:02d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1)
    tb = TensorBoard(log_dir='./logs')
    return [cp_callback, reduce_lr, tb]

def train(cfg, weights=None):
    X_train, y_train, X_val, y_val = load_data(cfg)
    models = load_model(config=cfg, weight_path=weights) 
    
    callback = call_callbacks()
    
    print('train')
    startTime1 = datetime.now() #DB
    hist1 = models.fit(X_train, y_train, epochs=cfg.epoch, batch_size=cfg.batch_size, validation_data=(X_val, y_val), callbacks=callback, verbose=1)

    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    _, acc = models.evaluate(X_val, y_val, verbose=0)
    print('\nTest accuracy: {0}'.format(acc))

if __name__=='__main__':
    cfg = get_option()
    os.makedirs(cfg.weight_dir, exist_ok=True)
    train(cfg, weights='pre_ep10.hdf5')
