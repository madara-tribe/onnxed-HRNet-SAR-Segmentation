import numpy as np
from sklearn.metrics import fbeta_score
from tensorflow.keras import backend as K
import tensorflow as tf

def eager_binary_fbeta(ytrue, ypred, beta=1.0, threshold=0.1):
  ypred = np.array(ypred >= threshold, dtype=np.float32)

  return fbeta_score(ytrue, ypred, beta)

def binary_fbeta(y_true, y_pred, beta=1, threshold=0.5, epsilon=1e-7):
    # epsilon is set so as to avoid division by zero error
    
    beta_squared = beta**2 # squaring beta

    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(y_true, tf.float32)
    ypred = tf.cast(y_pred, tf.float32)

    # setting values of ypred greater than the set threshold to 1 while those lesser to 0
    ypred = tf.cast(tf.greater_equal(y_pred, tf.constant(threshold)), tf.float32)

    tp = tf.reduce_sum(y_true*y_pred) # calculating true positives
    predicted_positive = tf.reduce_sum(y_pred) # calculating predicted positives
    actual_positive = tf.reduce_sum(y_true) # calculating actual positives
    
    precision = tp/(predicted_positive+epsilon) # calculating precision
    recall = tp/(actual_positive+epsilon) # calculating recall
    
    # calculating fbeta
    fb = (1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon)

    return fb
