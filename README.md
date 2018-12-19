# PnP-Depth
Implementation for "Plug-and-Play: Improve Depth Prediction via Sparse Data Propagation"

# Adding PnP module to your model
- Pytorch
```
import torch
from torch.autograd import Variable
from torch.autograd import grad as Grad

# Original inference
z = model.forward_front(inputs)
ori_pred = model.forward_rear(z)

# Inference with PnP
n_iters = 5 # number of iterations
alpha = 0.01 # update rate
z = model.forward_front(inputs)
for i in range(n_iters):
    if i != 0:
        z = z - alpha * torch.sign(z_grad) # iterative Fast Gradient Sign Method
    z = Variable(z, requires_grad=True)
    pred = model.forward_rear(z)
    if i < n_iters - 1:
        loss = criterion(pred, sparse_depth) # "sparse_depth" can be replaced with any partial ground truth
        z_grad = Grad([loss], [z], create_graph=True)[0]
# "pred" is the prediction after PnP module
```
- Tensorflow
```
import tensorflow as tf

def model_front(x):
    ... # whatever definition for your model here

def model_rear(z, reuse=False):
    with tf.variable_scope('', reuse=reuse):
        ... # whatever definition for your model here
        
# Original inference
z = model_front(inputs)
ori_pred = model_rear(z)

# Inference with PnP
n_iters = 5 # number of iterations
alpha = 0.01 # update rate

def _cond(z_loop, i_loop):
    return tf.less(i_loop, n_iters)

def _body(z_loop, i_loop):
    pred_loop = model_rear(z)
    loss_loop = criterion(pred_loop, sparse_depth) # "sparse_depth" can be replaced with any partial ground truth
    z_grad_loop = tf.gradients(loss_loop, z_loop)
    z_loop = tf.stop_gradients(z_loop - alpha * tf.sign(z_grad_loop))
    return z_loop, i_loop + 1
    
z = model_front(inputs)
z, _ = tf.while_loop(_cond, _body, (z, 0), back_prop=False, name='pnp')
pred = model_rear(z, True)
# "pred" is the prediction after PnP module
```
