import tensorflow as tf
import numpy as np

def maxpool(x, kern, stride):
    return tf.nn.max_pool(tf.pad(x, [[0, 0], [kern//2, kern//2],
                                      [kern//2, kern//2], [0, 0]]),
                          [ 1, kern, kern, 1 ], [ 1, stride, stride, 1], 'VALID')

def relu(x, leakness=0.0, name='relu'):
    if leakness > 0.0:
        return tf.maximum(x, x*leakness, name=name)
    else:
        return tf.nn.relu(x, name=name)

    
def sparse_conv(x, m, kern, out_filters, stride, name='sp_conv'):
    in_filters = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        sigsq = 2.0/(kern*kern*out_filters)
        kernel = tf.get_variable('kernel',
                                 [kern, kern, in_filters, out_filters],
                                 tf.float32,
                                 initializer = tf.random_normal_initializer(stddev= np.sqrt(sigsq)))
        bias = tf.get_variable('bias',
                               [1, 1, 1, out_filters],
                               tf.float32,
                               initializer = tf.zeros_initializer())
        sum_kernel = tf.ones(shape=[kern, kern, 1, 1])
        norm = tf.nn.conv2d(tf.pad(m, [[0, 0], [kern//2, kern//2],
                                       [kern//2, kern//2], [0, 0]]),
                            sum_kernel, [ 1, stride, stride, 1], 'VALID')
        x = tf.nn.conv2d(tf.pad(x * m, [[0, 0], [kern//2, kern//2],
                                        [kern//2, kern//2], [0, 0]]),
                           kernel, [ 1, stride, stride, 1], 'VALID') / (norm + 1e-8)
        x = x + bias
        m = maxpool(m, kern, stride)

    return x, m

def conv(x, kern_sz, out_filters, stride = 1, name='conv', use_bias = False):
    in_filters = x.get_shape().as_list()[-1]
    sigsq = 2.0/(kern_sz*kern_sz*out_filters)
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [kern_sz, kern_sz, in_filters, out_filters],
                                 tf.float32, initializer =
                                 tf.random_normal_initializer(stddev = np.sqrt(sigsq)))
        if use_bias:
            bias = tf.get_variable('bias',
                                   [1, 1, 1, out_filters],
                                   dtype = tf.float32,
                                   initializer = tf.zeros_initializer())
        else:
            bias = None
    if use_bias:
        out = tf.nn.conv2d(x, kernel, [ 1, stride, stride, 1 ], 'SAME') + bias
    else:
        out = tf.nn.conv2d(x, kernel, [ 1, stride, stride, 1 ], 'SAME')
    return out

def make_sparse_cnn(m1, d1, m2, d2):
    x, m = sparse_conv(d1, m1, 11, 16, 1, name = 'sp_conv1')
    x = relu(x)
    x, m = sparse_conv(x, m, 7, 16, 1, name = 'sp_conv2')
    x = relu(x)
    x, m = sparse_conv(x, m, 5, 16, 1, name = 'sp_conv3')
    x = relu(x)
    x, m = sparse_conv(x, m, 3, 16, 1, name = 'sp_conv4')
    x = relu(x)
    x, m = sparse_conv(x, m, 3, 16, 1, name = 'sp_conv5')
    x = relu(x)

    preds = conv(x, 1, 1)

    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(m2*(preds - d2), 2), axis = [1,2,3]))

    return preds, loss, {}, {}, None

##############################################################
##             Start of PnP-Depth modification              ##
##############################################################
def make_sparse_cnn_pnp(m1, d1, m2, d2):
    pnp_alpha = 0.01
    pnp_iters = 5

    x, m = sparse_conv(d1, m1, 11, 16, 1, name = 'sp_conv1') # x.shape = 16 x 512 x 1392 x 16, m.shape = 16 x 512 x 1392 x 1
    x = relu(x)
    first_x, first_m = x, m

    def model_rear(x, m, reuse=False):
        with tf.variable_scope('', reuse=reuse):
            x, m = sparse_conv(x, m, 7, 16, 1, name = 'sp_conv2') # x.shape = 16 x 512 x 1392 x 16, m.shape = 16 x 512 x 1392 x 1
            x = relu(x)
            x, m = sparse_conv(x, m, 5, 16, 1, name = 'sp_conv3') # ... all the same
            x = relu(x)
            x, m = sparse_conv(x, m, 3, 16, 1, name = 'sp_conv4')
            x = relu(x)
            x, m = sparse_conv(x, m, 3, 16, 1, name = 'sp_conv5') # x.shape = 16 x 512 x 1392 x 16
            x = relu(x)
            x = conv(x, 1, 1) # 16 x 512 x 1392 x 1
            return x

    def _cond(xadv_m, i):
        return tf.less(i, pnp_iters)

    def _body(xadv_m, i):
        pred_in = model_rear(*xadv_m)
        loss = tf.reduce_mean(tf.reduce_sum(m1*tf.abs(pred_in - d1), axis = [1,2,3]))
        grad = tf.gradients(loss, xadv_m[0])
        xadv_m[0] = tf.stop_gradient(xadv_m[0] - pnp_alpha*tf.sign(grad[0])*xadv_m[1])
        return xadv_m, i+1

    xadv = [first_x, first_m]
    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name='fast_gradient')
    preds = model_rear(*xadv, True)
    final_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(m2*(preds - d1), 2), axis = [1,2,3]))
    return preds, final_loss, {}, {}, None
##############################################################
##              End of PnP-Depth modification               ##
##############################################################
