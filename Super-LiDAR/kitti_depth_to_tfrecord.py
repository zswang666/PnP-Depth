import tensorflow as tf
import numpy as np
import os
import re
import functools
from PIL import Image
from itertools import accumulate
import pdb
import matplotlib.pyplot as plt
import dataloading as ld

def get_date(depth_filename):
    return re.search('2011_[0-9]{2}_[0-9]{2}', depth_filename).group()
def get_date_and_drive(depth_filename):
    return re.search('2011_[0-9]{2}_[0-9]{2}[^/]*', depth_filename).group()
def get_img_num(depth_filename):
    return int(re.search('([0-9]*).png', depth_filename).group(1))
def depth_path_to_img(depth_filename):
    date = get_date(depth_filename)
    date_and_drive = get_date_and_drive(depth_filename)
    image_dir = re.search('image_[0-9]{2}', depth_filename).group()
    image = os.path.basename(depth_filename)

    return functools.reduce(os.path.join, ['/dataset/kitti-depth/', date, date_and_drive,
                                           image_dir, 'data', image ])

def depth_path_to_raw(depth_filename):
    return re.sub('groundtruth', 'velodyne_raw', depth_filename)
            
def depth_selection_path_to_raw(depth_filename):
    return re.sub('groundtruth_depth', 'velodyne_raw', depth_filename)
def depth_selection_path_to_img(depth_filename):
    return re.sub('groundtruth_depth', 'image', depth_filename)

def get_train_paths(datapath):
    paths = []
    for root,dirs,files in os.walk(datapath):
        paths.extend([ os.path.join(root, file) for file in files ])
    return paths

def get_shards(root_dir, filter_re = None):
    paths = []
    for dir in os.listdir(root_dir):
        subpaths = []
        cur_root = os.path.join(root_dir, dir)
        for root,_,files in os.walk(cur_root):
            if filter_re is not None:
                subpaths.extend([ os.path.join(root, file) for file in files
                                  if re.search('groundtruth', root)
                                  and re.search(filter_re, root) ])
            else:
                subpaths.extend([ os.path.join(root, file) for file in files
                                  if re.search('groundtruth', root) ])
        subpaths.sort(key = get_img_num)
        paths.insert(0, subpaths)
    return paths
            
def get_shuffled_train_paths(datapath):
    paths = get_train_paths(datapath)
    order = np.random.permutation(len(paths))
    return [ paths[i] for i in order ]

def read_images(filename_bytes, depth_selection = False):
    filename = filename_bytes.decode()
    depth_png = np.expand_dims(np.array(Image.open(filename), dtype=np.int32),
                               axis = 2)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    if depth_selection:
        raw_filename = depth_selection_path_to_raw(filename)
    else:
        raw_filename = depth_path_to_raw(filename)
    raw_png = np.expand_dims(np.array(Image.open(raw_filename), dtype = np.int32),
                             axis = 2)
    assert(np.max(raw_png) > 255)
    #assert(np.sum(raw_png > 0) < np.sum(depth_png > 0))

    if depth_selection:
        image_filename = depth_selection_path_to_img(filename)
    else:
        image_filename = depth_path_to_img(filename)

    img_png = np.array(Image.open(image_filename), dtype=np.int32)

    #rgb = tf.constant(img_png, dtype=tf.int32)
    #d = tf.constant(depth_png, dtype=tf.int32)
    return img_png, depth_png, raw_png

def encode_images(rgb, d, raw):
    rgb_png = tf.image.encode_png(tf.cast(rgb, dtype=tf.uint8))
    d_png = tf.image.encode_png(tf.cast(d, dtype=tf.uint16))
    raw_png = tf.image.encode_png(tf.cast(raw, dtype=tf.uint16))
    return rgb_png, d_png, raw_png

def make_record(rgb_bytes, d_bytes, raw_bytes, seq_id):
    ex = tf.train.Example(features = tf.train.Features(feature = {
        'rgb_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb_bytes])),
        'd_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[d_bytes])),
        'raw_bytes' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_bytes])),
        'seq_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seq_id.encode()]))
    }))
    return ex.SerializeToString()


def convert(filenames, output_file, sess, depth_selection):
    if os.path.isfile(output_file):
        print('Skipping {}, file already exists'.format(output_file))
    else:
        print('Writing {} files to {}'.format(len(filenames), output_file))
        files_datset = tf.data.Dataset.from_tensor_slices(filenames)
        parsed = files_datset.map(lambda filename: tuple(tf.py_func(
            lambda x : read_images(x, depth_selection),
            [filename], [tf.int32, tf.int32, tf.int32])), num_parallel_calls=4)
        parsed = parsed.prefetch(100)
        encoded = parsed.map(encode_images, num_parallel_calls=4)
        zipped = tf.data.Dataset.zip((files_datset, encoded))

        it = zipped.make_one_shot_iterator()
        filename_t, (rgb_t, d_t, raw_t) = it.get_next()
        with tf.python_io.TFRecordWriter(output_file) as writer:
            i = 1
            while True:
                try:
                    filename, rgb_png, depth_png, raw_png = sess.run([filename_t, rgb_t,
                                                                      d_t, raw_t])
                except tf.errors.OutOfRangeError:
                    break
                if i % 100 == 0:
                    print('wrote {}'.format(i))
                seq_id = filename.decode()
                pinrt(seq_id)
                print(get_img_num(filename.decode()))
                ex = make_record(rgb_png, depth_png, raw_png, seq_id)
                writer.write(ex)
                i = i + 1
def convert_depth_selection(root_dir, output_file):
    files = get_train_paths(root_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        convert(files, output_file, sess, depth_selection = True)

def convert_dataset(root_dir, output_dir):
    #filenames = get_train_paths(root_dir)

    shardnum = 1
    print('Outputting to {}, all files will be overwritten'.format(output_dir))
    input("Press Enter to continue...")
    
    shard_filenames = get_shards(root_dir, filter_re = 'image_02')
    print(shard_filenames)
    
    numfiles = sum([ len(s) for s in shard_filenames])
    print('Writing {} files to {} shards'.format(numfiles, len(shard_filenames)))

    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        for shard in    shard_filenames:
            output_file = os.path.join(output_dir,
                                       '{}.tfrecords'.format(get_date_and_drive(shard[0])))
            convert(shard, output_file, sess, depth_selection = False)
            shardnum += 1

def make_small_dataset(record_dir, output_file, size):
    files = ld.get_train_paths(record_dir)
    num_examples = ld.count_records(files)
    
    s = np.random.choice(num_examples, size, replace = False)
    filt = np.zeros(num_examples)
    filt[s] = 1

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    take_pl = tf.placeholder(shape=(None), dtype=tf.int64)
    
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(lambda x : tf.data.TFRecordDataset(x), cycle_length = 1)
    dataset = dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(take_pl)))
    dataset = dataset.filter(lambda x, i: tf.greater(i, 0)).map(lambda x, i: x).batch(1)




    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    data_init_op = iterator.make_initializer(dataset)
    ex_str = iterator.get_next()

    sess.run(data_init_op, feed_dict = { take_pl : filt })
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for i in range(num_examples):
            ex = sess.run(ex_str)
            writer.write(ex[0])
            if i % 50 == 0:
                print('Wrote {}'.format(i))
        

convert_dataset('/path/to/your/data/train/'
                '/path/to/your/output/directory/')
