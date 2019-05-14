import tensorflow as tf
import numpy as np

def relu(x, leakness=0.0, name='relu'):
    if leakness > 0.0:
        return tf.maximum(x, x*leakness, name=name)
    else:
        return tf.nn.relu(x, name=name)

def bn(x, is_training, name='bn'):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x, momentum = 0.9,
                                             center = True, scale = True,
                                             training = is_training)

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

def upproj(x, out_depth, is_training, name='upproj', use_batchnorm = True):
    with tf.variable_scope(name) as scope:
        x = unpool(x)
        shortcut = conv(x, 5, out_depth, 1, name='shortcut_conv',
                        use_bias = not use_batchnorm)
        if use_batchnorm:
            shortcut = bn(shortcut, is_training, name='shortcut_bn')
        
        x = conv(x, 5, out_depth, 1, name='conv1', use_bias = not use_batchnorm)
        if use_batchnorm:
            x = bn(x, is_training, name='bn1')
        x = relu(x, name='relu1')
        x = conv(x, 3, out_depth, 1, use_bias = not use_batchnorm)
        if use_batchnorm:
            x = bn(x, is_training, name='bn2')

        x = relu(x + shortcut, name='relu2')

    return x
def shortcut(x, nInput, nOutput, stride, is_training,
             name='shortcut', use_batchnorm = True, use_bias = False):
    if nInput != nOutput:
        with tf.variable_scope(name):
            x = conv(x, 1, nOutput, stride, name='conv', use_bias = use_bias)
            if use_batchnorm:
                x = bn(x, is_training, name='bn')
        return x;
    else:
        return x;

def basicblock(x, n, stride, is_training, name='basicblock',
               use_batchnorm = True, use_bias = False):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        cut = shortcut(x, in_channel, n, stride, is_training,
                       use_bias = use_bias,
                       use_batchnorm = use_batchnorm)

        x = conv(x, 3, n, stride, name='conv1', use_bias = use_bias)
        if use_batchnorm:
            x = bn(x, is_training, name='bn1')
        x = relu(x, name='reul1')
        x = conv(x, 3, n, 1, name='conv2', use_bias = use_bias)
        if use_batchnorm:
            x = bn(x, is_training, name='bn2')

        x = x + cut
        x = relu(x, name='relu2')
    return x

def unpool(x):
    xshape = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    filt = np.zeros([2, 2, xshape[-1], xshape[-1]])
    for i in range(xshape[-1]):
        filt[0, 0, i, i] = 1

    filt_tens = tf.constant(filt, dtype=tf.float32)
    out = tf.nn.conv2d_transpose(x, filt_tens, tf.stack([ batch_size, 2*xshape[1],
                                                          2*xshape[2], xshape[3] ]),
                                 [1, 2, 2, 1], 'VALID')
    return out
def maxpool(x, kern, stride):
    return tf.nn.max_pool(tf.pad(x, [[0, 0], [kern//2, kern//2],
                                      [kern//2, kern//2], [0, 0]]),
                          [ 1, kern, kern, 1 ], [ 1, stride, stride, 1], 'VALID')


def build_net18(m1, d1, m2, d2, is_training):
    block_sizes = [ 2, 2, 2, 2 ]
    block_filters = [32, 64, 128, 256]
    block_strides = [ 1, 2, 2, 2 ]
    use_batchnorm = False
    with tf.variable_scope('block0') as scope:
        x = conv(d1, 7, 16, 2, name='conv1', use_bias = not use_batchnorm)
        if use_batchnorm:
            x = bn(x, is_training)
        x = relu(x)
        x = maxpool(x, 3, 2)

    blockno = 1
    for size, filters, stride in zip(block_sizes, block_filters,
                                     block_strides):
        print('Making basic block {}'.format(blockno))
        with tf.variable_scope('block{}'.format(blockno)) as scope:
            for i in range(size):
                x = basicblock(x, filters, stride if i == 0 else 1,
                               is_training, name='basicblock{}'.format(i+1),
                               use_batchnorm = use_batchnorm,
                               use_bias = not use_batchnorm)
            blockno = blockno + 1

    with tf.variable_scope('bridge'):
         x = conv(x, 1, block_filters[-1]/2, 1, use_bias = not use_batchnorm)
         if use_batchnorm:
             x = bn(x, is_training)

    out_channel = block_filters[-1]/4
    num_upproj = 1 + sum([1 if stride > 1 else 0 for stride in block_strides])
    for i in range(num_upproj):
        with tf.variable_scope('upproj{}'.format(i+1)):
            x = upproj(x, out_channel, is_training, use_batchnorm=use_batchnorm)
        out_channel = out_channel // 2
    with tf.variable_scope('final'):
        x = conv(x, 3, d2.get_shape().as_list()[-1], 1)
    preds = tf.image.resize_images(x, tf.shape(d2)[1:3])

    wd_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                         if 'kernel'])*0.004
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(m2*(preds - d2), 2), axis = [1,2,3])) + wd_loss
    return preds, loss, {}, {}, None

