import sys
import tensorflow as tf
import dataloading as ld
import losses
import numpy as np
import time
from errors import ErrorLogger
import os
import pdb
import pickle
import argparse
from deep_cnn import build_net18
import admm
from sparse_cnn import make_sparse_cnn
from PIL import Image
import re
## Inputs

def sparsify(x, m, prob):
    mask = tf.distributions.Bernoulli(probs = tf.fill(tf.shape(x)[0:2], prob),
                                      dtype = tf.bool).sample()
    mask = tf.expand_dims(mask, 2)
    mask = tf.tile(mask, [1, 1, x.get_shape()[2]])
    mask = tf.cast(tf.logical_and(mask, tf.greater(m, 0)), tf.float32)
    return mask, mask*x

## TRAINING

def main(result_dir, resume_file, resume_epoch, nepochs, f, input_type, model_type,
         num_iters, admm_filters, admm_strides, admm_kernels, lr, val_only, train_size,
         val_size, dataset, redraw_subset, batch_size, repeat, admm_tv_loss, no_vis_output,
         val_output_every, png_output, png_output_dir):
    def clip(rgb):
        return np.maximum(np.minimum(rgb, 255), 0)

    if dataset == 'kitti':

        trainfiles = ld.get_train_paths('/dataset/kitti-depth/tfrecords/train')
        num_train_examples = ld.count_records(trainfiles)

        print('Got {} training files with {} records'.format(len(trainfiles), num_train_examples))

        valfiles = ld.get_train_paths('/dataset/kitti-depth/tfrecords/val')
        num_val_examples = ld.count_records(valfiles)
        print('Got {} validation files with {} records'.format(len(valfiles), num_val_examples))
        make_datasets = lambda mkinpts, bs: ld.make_kitti_datasets(mkinpts, trainfiles, valfiles,
                                                                   bs, repeat = repeat)
    elif dataset == 'kitti_test_selection':
        test_root = '/dataset/kitti-depth/depth_selection/test_depth_completion_anonymous'
        num_train_examples = len(ld.get_train_paths(test_root + '/velodyne_raw', suffix='png'))
        num_val_examples = num_train_examples
        make_datasets = lambda mkinpts, bs : ld.make_selection_datasets(mkinpts, test_root)
    elif dataset == 'kitti_val_selection':
        val_root = '/dataset/kitti-depth/depth_selection/val_selection_cropped'
        num_train_examples = len(ld.get_train_paths(val_root + '/velodyne_raw', suffix='png'))
        num_val_examples = num_train_examples
        make_datasets = lambda mkinpts, bs : ld.make_selection_datasets(mkinpts, val_root)

    print('Got {} training examples'.format(num_train_examples))
    print('Got {} validation examples'.format(num_val_examples))

    if train_size < 0:
        train_size = num_train_examples
    if val_size < 0:
        val_size = num_val_examples
            
    if input_type == 'raw':
        def make_raw_inputs(urgb, m, g, mraw, raw, s):
            m1 = mraw
            return urgb, m1, m1 * raw, m, g, s
        make_inputs = make_raw_inputs
    elif input_type == 'raw_frac':
        def make_raw_frac_inputs(urgb, m, g, mraw, raw, s):
            m1, d1 = sparsify(raw, mraw, f)
            return urgb, m1, d1, m, g, s
        make_inputs = make_raw_frac_inputs
        
    if model_type == 'admm':
        def build_admm(m1, d1, m2, d2, is_training):
            return admm.make_admm(m1, d1, m2, d2,
                                  tv_loss = admm_tv_loss,
                                  num_iters = num_iters, filters = admm_filters,
                                  strides = admm_strides, kernels = admm_kernels)
        build_model = build_admm
    elif model_type == 'cnn_deep':
        build_model = lambda m1, d1, m2, d2, is_training : build_net18(m1, d1, m2, d2, is_training)
    elif model_type == 'sparse_cnn':
        build_model = lambda m1, d1, m2, d2, is_training : make_sparse_cnn(m1, d1, m2, d2)
    
    train_log = os.path.join(result_dir, 'train_log.txt')
    train_errors = ErrorLogger(['rmse', 'grmse', 'mae', 'gmae', 'mre',
                                'del_1', 'del_2', 'del_3', ],
                               [(8,5), (8,5), (8,5), (8,5), (8,5),
                                (5,2), (5,2), (5,2)], train_log)
    val_log = os.path.join(result_dir, 'val_log.txt')
    val_errors = ErrorLogger(['rmse', 'grmse', 'mae', 'gmae', 'mre',
                                'del_1', 'del_2', 'del_3', ],
                               [(8,5), (8,5), (8,5), (8,5), (8,5),
                                (5,2), (5,2), (5,2)], val_log)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:

        train_dataset, val_dataset, take_pl = make_datasets(make_inputs, batch_size)
        print(train_dataset.output_shapes)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
        rgb_t, m1_t, d1_t, ground_mask, ground, s_t = iterator.get_next()
        
        train_data_init_op = iterator.make_initializer(train_dataset)
        val_data_init_op = iterator.make_initializer(val_dataset)

        is_training = tf.placeholder(tf.bool, name='is_training')
        output, loss, monitor, summary, model_train_op = build_model(m1_t, d1_t,
                                                                     ground_mask, ground,
                                                                     is_training)
        
        mse_t = losses.mse_loss(output, ground, ground_mask)
        mae_t = losses.mae_loss(output, ground, ground_mask)
        mre_t = losses.mre_loss(output, ground, ground_mask)
        rmse_t = losses.rmse_loss(output, ground, ground_mask)
        gmae_t = losses.mae_loss(output, ground, m1_t * ground_mask)
        grmse_t = losses.rmse_loss(output, ground, m1_t * ground_mask)
        del_1_t, del_2_t, del_3_t = losses.deltas(output, ground, ground_mask, 1.01)
        
        errors_t = { 'rmse' : rmse_t, 'mae' : mae_t, 'mre' : mre_t,
                     'del_1' : del_1_t, 'del_2' : del_2_t, 'del_3' : del_3_t,
                     'grmse' : grmse_t, 'gmae' : gmae_t}

        
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        if model_train_op is not None:
            train_op = model_train_op
        else:
            extra_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_train_op):
                train_op = optimizer.minimize(loss)

        saver = tf.train.Saver(max_to_keep = nepochs + 1)
        sess.run(tf.global_variables_initializer())
        if resume_file:
            print('Restoring from {}'.format(resume_file))
            saver.restore(sess, resume_file)

        best_rmse = float('inf')
        best_epoch = -1

        train_take = ld.make_take(num_train_examples, train_size)
        val_take = ld.make_take(num_val_examples, val_size)
        
        num_epochs = nepochs
        if val_only:
            num_epochs = 1
        
        for i in range(resume_epoch, num_epochs):
            if not val_only:
                num_batches = train_size // batch_size
                batchnum = 1

                if redraw_subset:
                    print('Redrawing Subset')
                    train_take = ld.make_take(num_train_examples, train_size)
                train_errors.clear()
                sess.run(train_data_init_op, feed_dict = { take_pl : train_take })
                while True: 
                    try:
                        start = time.time()
                        (err, pred, mg, g, rgb,
                         m1, d1, m, s, _) = sess.run([errors_t, output,
                                                      ground_mask, ground,
                                                      rgb_t,
                                                      m1_t, d1_t,
                                                      monitor, summary,
                                                      train_op],
                                                     feed_dict = { is_training : True})
                        print('{}s to run'.format(time.time() - start))
                        train_errors.update(err)
                        print('{} in input, {} in ground truth'.
                              format(np.mean(np.sum(m1 > 0, axis = (1,2,3))),
                                     np.mean(np.sum(mg > 0, axis = (1,2,3)))))
                        print('Epoch {}, Batch {}/{} {}'.
                              format(i, batchnum, num_batches,
                                     train_errors.update_log_string(err)))
                        for key, value in m.items():
                            print('{}: {}'.format(key, value))
                        if batchnum % 500 == 0:
                            filename = 'train_output{}.pickle'.format(batchnum)
                            with open(os.path.join(result_dir, filename), 'wb') as f:
                                pickle.dump({ 'rgb' : clip(rgb[0, :, :, :]),
                                              'd1' : m1[0, :, :, :]*d1[0, :, :, :],
                                              'm0' : s['m'][0] if 'm' in s else None,
                                              'ground' : g[0, :, :, :],
                                              'pred' : pred[0, :, :, :],
                                              'summary' : s }, f)
                        batchnum += 1

                    except tf.errors.OutOfRangeError:
                        break
                train_errors.log()
                with open(os.path.join(result_dir, 'summary.pickle'), 'wb') as f:
                    pickle.dump(s, f)
                print('Done epoch {}, RMSE = {}'.format(i, train_errors.get('rmse')))
                save_path = saver.save(sess, os.path.join(result_dir, '{:02}-model.ckpt'.format(i)))
                print('Model saved in {}'.format(save_path))

            num_batches = val_size
            batchnum = 1

            val_errors.clear()
            sess.run(val_data_init_op, feed_dict = { take_pl : val_take })
            best_batch = float('inf')
            worst_batch = 0
            rmses = {}
            i = 0

            while True:
                try:
                    start = time.time()
                    (err, pred, g,
                     rgb, m1, d1, m, s, seqid) = sess.run([errors_t, output,
                                                           ground, rgb_t,
                                                           m1_t, d1_t, monitor, summary,
                                                           s_t],
                                                          feed_dict = { is_training : False })
                    print('{}s to run'.format(time.time() - start))
                    rmses[i] = err['rmse']
                    i = i + 1
                    val_errors.update(err)
                    print('{}/{} {}'.format(batchnum, num_batches,
                                            val_errors.update_log_string(err)))
                    for key, value in m.items():
                        print('{}: {}'.format(key, value))
                    if png_output:
                        ID = os.path.basename(seqid[0].decode())
                        filename = os.path.join(png_output_dir, ID)
                        out = np.round(np.squeeze(pred[0, :, :, 0])*256.0);
                        out = out.astype(np.int32)
                        Image.fromarray(out).save(filename, bits=16)
                        
                    if not no_vis_output:
                        vis_log = { 'rgb' : rgb[0, :, :, :],
                                    'd1' : m1[0, :, :, :]*d1[0, :, :, :],
                                    'ground' : g[0, :, :, :],
                                    'pred' : pred[0, :, :, :] }
                        if 'm' in s:
                            vis_log['m0'] = s['m'][0]
                        if err['rmse'] < best_batch:
                            best_batch = err['rmse']
                            filename = os.path.join(result_dir,
                                                    'val_best.pickle')
                            with open(filename, 'wb') as f:
                                pickle.dump(vis_log, f)
                        if err['rmse'] > worst_batch:
                            worst_batch = err['rmse']
                            filename = os.path.join(result_dir,
                                                    'val_worst.pickle')
                            with open(filename, 'wb') as f:
                                pickle.dump(vis_log, f)
                        if batchnum % val_output_every == 0:
                            filename = os.path.join(result_dir,
                                                    'val_output-{:04}.pickle'.format(batchnum))
                            with open(filename, 'wb') as f:
                                pickle.dump(vis_log, f)
                    batchnum += 1
                except tf.errors.OutOfRangeError:
                    break
            val_errors.log()
            if val_errors.get('rmse') < best_rmse and not val_only:
                best_epoch = i
                best_rmse = val_errors.get('rmse')
                save_path = saver.save(sess, os.path.join(result_dir, 'best-model.ckpt'))
                print('Best model saved in {}'.format(save_path))
            with open(os.path.join(result_dir, 'errors.pickle'), 'wb') as f:
                pickle.dump(rmses, f)
            print('Validation RMSE: {}'.format(val_errors.get('rmse')))

parser = argparse.ArgumentParser()
parser.add_argument('dir', help = 'the directory to store all of the output')
parser.add_argument('--type', help = 'the type of model to use',
                    choices = ['admm', 'cnn_deep', 'sparse_cnn'],
                    default = 'admm')
parser.add_argument('--input', help = ("the structure of the model input (usually ortho for"
                                       "admm and subset for cnn"),
                    default = 'raw', choices = ['raw', 'raw_frac'])

parser.add_argument('--frac',
                    help = 'the fraction of samples to include as input for the raw_frac input',
                    type = float, default = 0.5)

parser.add_argument('--num_iters',
                    help = 'the number of admm iterations to perform',
                    type = int, default = 10)

parser.add_argument('--admm_filters',
                    help = 'the number of filters for the admm or cnn to learn',
                    type = int, nargs = '+', default = [ 8, 16, 32 ] )
parser.add_argument('--admm_strides',
                    help = 'the stride of the admm or cnn convolutions',
                    type = int, nargs = '+', default = [ 2, 2, 2 ])
parser.add_argument('--admm_kernels',
                    help = 'the kernel sizes for the admm layers',
                    type = int, nargs = '+', default = [ 11, 5, 3 ])

parser.add_argument('--admm_tv_loss',
                    help = ('the weight given to the total variation loss for admm output,'
                            'if None then the no TV loss is used'),
                    default = 0.1, type = float)

parser.add_argument('--resume_file',
                    help = ('the checkpoint file to resume from,'
                            'if not given model is trained from scratch'),
                    default = None)
parser.add_argument('--resume_epoch',
                    help = ('the epoch number to start at,'
                            'useful when resuming part way through training'),
                    default = 0, type = int)

parser.add_argument('--learning_rate',
                    help = 'the learning rate for the ADAM optimizer',
                    default = 0.001, type = float)

parser.add_argument('--val_only',
                    help = 'only run validation with no training',
                    default = False, action = 'store_true')

parser.add_argument('--val_size',
                    help = 'the number of validation examples to test (-1 for all)',
                    default = -1, type = int)

parser.add_argument('--train_size',
                    help = 'the number of train examples to use (-1 for all)',
                    default = -1, type = int)

parser.add_argument('--dataset',
                    help = 'the dataset to train on',
                    default = 'kitti', choices = ['kitti', 'kitti_test_selection',
                                                  'kitti_val_selection'])

parser.add_argument('--num_epochs',
                    help = 'The number of epochs to train for',
                    default = 6, type = int)

parser.add_argument('--dont_redraw_subset',
                    help = 'If given, redraw the training subset before each epoch',
                    action = 'store_false')

parser.add_argument('--batch_size',
                    help = 'the batch size',
                    default = 16, type = int)

parser.add_argument('--repeat_dataset',
                    help = 'the number of times to repeat a dataset before running validation',
                    default = 1, type = int)

parser.add_argument('--no_vis_output',
                    help = 'turn off writing pickle files of visual outputs',
                    default = False, action = 'store_true')

parser.add_argument('--val_output_every',
                    help = 'the interval in between successive validation outputs',
                    default = 500, type = int)

parser.add_argument('--png_output',
                    help = ('if given then validation predictions will be written'
                            ' to png files for evaluation'),
                    action = 'store_true')
parser.add_argument('--png_output_dir',
                    help = 'the directory to store png outputs',
                    default = 'pngs')

args = parser.parse_args()

main(args.dir, resume_file = args.resume_file, resume_epoch = args.resume_epoch,
     f = args.frac, input_type = args.input, model_type = args.type,
     num_iters = args.num_iters, admm_filters = args.admm_filters, admm_strides=args.admm_strides,
     admm_kernels = args.admm_kernels,
     lr = args.learning_rate, val_only = args.val_only, val_size = args.val_size,
     train_size = args.train_size, dataset = args.dataset, nepochs = args.num_epochs,
     redraw_subset = args.dont_redraw_subset, batch_size = args.batch_size,
     repeat = args.repeat_dataset,
     admm_tv_loss = args.admm_tv_loss, no_vis_output = args.no_vis_output,
     val_output_every = args.val_output_every,
     png_output = args.png_output, png_output_dir = args.png_output_dir)
