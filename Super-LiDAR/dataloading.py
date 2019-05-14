import os
import time
import threading
import h5py as h5
from skimage.transform import resize
import scipy.io
from scipy.ndimage import rotate
import numpy as np
import math
import tensorflow as tf
import pdb
import re
import pickle


def get_train_paths(datapath, suffix = '.tfrecords'):
    paths = []
    for root,dirs,files in os.walk(datapath, followlinks=True):
        paths.extend([ os.path.join(root, file) for file in files if re.search(suffix+'$', file)])
    return paths

def get_shuffled_train_paths(datapath):
    paths = get_train_paths(datapath)
    order = np.random.permutation(len(paths))
    return [ paths[i] for i in order ]



def kitti_parse_function(ex_str):
    keys = { 'rgb_bytes': tf.VarLenFeature(tf.string),
             'd_bytes': tf.VarLenFeature(tf.string),
             'raw_bytes' : tf.VarLenFeature(tf.string),
             'seq_id': tf.VarLenFeature(tf.string)}
    features = tf.parse_single_example(ex_str, features=keys)
    rgb = tf.cast(tf.image.decode_png(tf.reshape(tf.sparse_tensor_to_dense(features['rgb_bytes'],
                                                                           default_value=''),
                                                 ()),
                                      channels=3),
                  tf.float32)
    ground = tf.cast(tf.image.decode_png(tf.reshape(tf.sparse_tensor_to_dense(features['d_bytes'],
                                                                              default_value=''),
                                                    ()),
                                         channels=1, dtype=tf.uint16),
                tf.float32)
    ground = tf.squeeze(ground)
    raw = tf.cast(tf.image.decode_png(tf.reshape(tf.sparse_tensor_to_dense(features['raw_bytes'],
                                                                           default_value=''),
                                                 ()),
                                      channels=1, dtype=tf.uint16),
                tf.float32)
    raw = tf.squeeze(raw)
    return rgb, ground, raw, tf.sparse_tensor_to_dense(features['seq_id'], default_value='')


# Use standard TensorFlow operations to normalize the rgb and depth images
def kitti_normalize_function(rgb, ground, raw, seqid):
    #rgb = tf.transpose(rgb, perm=[1, 2, 0])
    rgb = rgb[0:370, 0:1220, :]
    rgb.set_shape([370, 1220, 3])

    ground = ground/256.0
    ground = ground[0:370, 0:1220]
    ground.set_shape([370, 1220])

    raw = raw/256.0
    raw = raw[0:370, 0:1220]
    raw.set_shape([370, 1220])
    return rgb, ground, raw, seqid

# Use standard TensorFlow operations to augment the training data
def kitti_augment_function(rgb, ground, raw, seqid):
    degree = tf.random_uniform((), minval=-2.5, maxval=2.5)*math.pi/180
    s = tf.random_uniform((), 1.0, 1.5)
    flip = tf.greater(tf.random_uniform((), 0, 1), 0.5)

    
    ground = tf.contrib.image.rotate(ground, degree)
    ground = ground/s
    ground = tf.cond(flip, lambda: ground, lambda: ground[:, ::-1])

    raw = tf.contrib.image.rotate(raw, degree)
    raw = raw/s
    raw = tf.cond(flip, lambda: raw, lambda: raw[:,::-1])
    return rgb, ground, raw, seqid


def make_interleaved_dataset(records, parse, shuffle=None, take=None):
    dataset = tf.data.Dataset.from_tensor_slices(records)
    dataset = dataset.interleave(lambda x : tf.data.TFRecordDataset(x),
                                 cycle_length = tf.cast(tf.reduce_prod(tf.shape(records)),
                                                        tf.int64),
                                 block_length = 4)
    if take is not None:
        dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(take)))
        dataset = dataset.filter(lambda x, i: tf.greater(i, 0)).map(lambda x, i: x)
    if shuffle is not None:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.prefetch(50)
    dataset = dataset.map(parse, num_parallel_calls = 4)
    return dataset
def make_train_dataset(filenames, parse, norm, aug,
                       shuffle, take = None, repeat = 1):
    dataset = make_interleaved_dataset(filenames, parse, shuffle, take)
    dataset = dataset.repeat(repeat)
    dataset = dataset.prefetch(50)    
    dataset = dataset.map(norm, num_parallel_calls = 4)
    dataset = dataset.map(aug, num_parallel_calls = 4)
    dataset = dataset.prefetch(50)
    return dataset

def make_val_dataset(filenames, parse, norm, take = None):
    dataset = make_interleaved_dataset(filenames, parse, shuffle=None, take=take)
    dataset = dataset.prefetch(16)
    dataset = dataset.map(norm, num_parallel_calls = 4)
    return dataset

def count_examples(record):
    realpath = os.path.realpath(record)
    picklepath = re.sub('\.tfrecords', '.pickle', realpath)
    if os.path.isfile(picklepath):
        with open(picklepath, 'rb') as f:
            meta = pickle.load(f)
        return meta['numexamples']
    else:
        count = sum([1 for i in tf.python_io.tf_record_iterator(record)])
        with open(picklepath, 'wb') as f:
            meta = { 'numexamples' : count }
            pickle.dump(meta, f)
        return count

def count_records(records):
    return sum([count_examples(record) for record in records]) 

def make_selection_datasets(make_inputs, root_dir):
    raw_images = get_train_paths(os.path.join(root_dir,'velodyne_raw'), suffix='png')
    def raw_to_image_filename(f):
        return re.sub('velodyne_raw', 'image', f)
    def raw_to_groundtruth_filename(f):
        return re.sub('velodyne_raw', 'groundtruth_depth', f)
    def parse_depth(filename):
        filecontents = tf.read_file(filename)
        png = tf.cast(tf.image.decode_png(filecontents, dtype=tf.uint16, channels=1), tf.float32)
        png = tf.squeeze(png)
        return png
    def parse_rgb(filename):
        filecontents = tf.read_file(filename)
        png = tf.image.decode_png(filecontents, dtype=tf.uint8)
        return png
    def munge_data(rgb, ground, raw, s):
        ground = tf.expand_dims(ground, 2)
        raw = tf.expand_dims(raw, 2)        
        m = tf.cast(tf.greater(ground, 0), tf.float32)
        mraw = tf.cast(tf.greater(raw, 0), tf.float32)
        return (rgb, m, ground, mraw, raw, s)
    
    raw = tf.data.Dataset.from_tensor_slices(raw_images).map(parse_depth)
    rgb = tf.data.Dataset.from_tensor_slices([ raw_to_image_filename(f)
                                              for f in raw_images ]).map(parse_rgb)
    if os.path.isdir(os.path.join(root_dir, 'groundtruth_depth')):
        print('Found groundtruth')
        ground = tf.data.Dataset.from_tensor_slices([ raw_to_groundtruth_filename(f)
                                                      for f in raw_images ]).map(parse_depth)
    else:
        print('Groundtruth not found! Evaluation metrics will be innacurate')
        ground = raw
    d = tf.data.Dataset.zip((rgb, ground, raw, tf.data.Dataset.from_tensor_slices(raw_images)))
    d = d.prefetch(50)
    d = d.map(kitti_normalize_function).map(munge_data).map(make_inputs).batch(1)

    take_pl = tf.placeholder(shape=(None), dtype=tf.int64)
    return d, d, take_pl
    
def make_kitti_datasets(make_inputs, trainfiles, valfiles, batch_size, repeat):
    def munge_data(rgb, ground, raw, s):
        ground = tf.expand_dims(ground, 2)
        raw = tf.expand_dims(raw, 2)
        m = tf.cast(tf.greater(ground, 0), tf.float32)
        mraw = tf.cast(tf.greater(raw, 0), tf.float32)
        return (rgb, m, ground, mraw, raw, s)
    take_pl = tf.placeholder(shape=(None), dtype=tf.int64)

    train_dataset = make_train_dataset(trainfiles,
                                       kitti_parse_function,
                                       kitti_normalize_function,
                                       kitti_augment_function,
                                       shuffle = 3500, take=take_pl, repeat = repeat)
    train_dataset = train_dataset.map(munge_data)
    train_dataset = train_dataset.map(make_inputs)
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = make_val_dataset(valfiles,
                                   kitti_parse_function,
                                   kitti_normalize_function,
                                   take=take_pl)
    val_dataset = val_dataset.map(munge_data)
    val_dataset = val_dataset.map(make_inputs)
    val_dataset = val_dataset.batch(1)

    return train_dataset, val_dataset, take_pl


def make_take(total, sample):
    s = np.random.choice(total, sample, replace = False)
    filt = np.zeros(total)
    filt[s] = 1
    return filt
