import tensorflow as tf
import numpy as np

def l1_loss(preds, tars):
    mask = tf.greater(tf.abs(tars), 0)
    residuals = tf.boolean_mask(tars - preds, mask)
    mae = tf.reduce_mean(tf.abs(residuals))
    return mae

def mse_loss(preds, tars, mask):
    residuals = tf.boolean_mask(tars - preds, tf.greater(mask, 0))
    mse = tf.reduce_mean(tf.pow(residuals, 2))
    return mse

def mae_loss(preds, tars, mask):
    residuals = tf.boolean_mask(tars - preds, tf.greater(mask, 0))
    mae = tf.reduce_mean(tf.abs(residuals))
    return mae

def rmse_loss(preds, tars, mask):
    counts = tf.reduce_sum(mask, axis=[1,2,3], keep_dims=True)
    errors = tf.reduce_sum(tf.pow((tars - preds)*mask, 2), axis=[1,2,3], keep_dims=True)
    return tf.reduce_mean(tf.sqrt(errors/counts))

def mre_loss(preds, tars, mask):
    residuals = tf.boolean_mask(tars - preds, tf.greater(mask, 0))
    tars_masked = tf.boolean_mask(tars, tf.greater(mask, 0))
    return tf.reduce_mean(tf.abs(residuals/(tars_masked + 1e-6)))

def given_l1_loss(preds, images):
    given = images[:, :, :, 3]
    given_mae = l1_loss(preds, tf.expand_dims(given, 3))
    return given_mae

def weight_decay_loss():
    wd_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                          if 'kernel']) * 0.0001
    return wd_loss

def deltas(preds, tars, mask, thresh):
    preds_masked = tf.boolean_mask(preds, tf.greater(mask, 0))
    tars_masked = tf.boolean_mask(tars, tf.greater(mask, 0))
    rel = tf.maximum(preds_masked/tars_masked, tars_masked/(preds_masked+1e-3))
    N = tf.reduce_sum(mask)
    def del_i(i):
        return tf.reduce_mean(tf.cast(tf.less(rel, thresh ** i), tf.float32))
    return del_i(1), del_i(2), del_i(3)

def del_i(preds_arr, tars_arr, thresh):
    mask = np.abs(tars_arr) > 0
    rel = np.maximum(preds_arr[mask]/tars_arr[mask], tars_arr[mask]/preds_arr[mask])
    N = np.sum(mask)
    return np.sum(rel < thresh)/N, np.sum(rel < thresh ** 2)/N, np.sum(rel < thresh ** 3)/N

def scale(preds, tars):
    mask = tf.cast(tf.greater(tf.abs(tars), 0), tf.float32)
    s = (tf.reduce_sum(preds * tars, axis=[1, 2, 3], keep_dims=True) /
         tf.reduce_sum(preds * preds * mask, axis=[1,2,3], keep_dims=True))
    return s*preds
def scale_inv_l2_loss(preds, tars):
    mask = tf.cast(tf.greater(tf.abs(tars), 0), tf.float32)
    spreds = scale(preds, tars)
    return tf.reduce_mean(tf.pow((spreds - tars)*mask, 2))
    

