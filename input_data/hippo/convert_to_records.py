# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import re
import scipy.misc
import skimage.io as io
import numpy as np

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_data():
    images = []
    labels = []
    data_dirs = FLAGS.dirs.split()
    for dir_index in range(len(data_dirs)):
        data_dir = data_dirs[dir_index]
        print('Reading', data_dir)
        for f in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, f)):
                number_search = re.search(r'hippo(\d+)\.png', f)
                if number_search:
                    number = number_search.group(1)
                    # image = scipy.misc.imread(os.path.join(FLAGS.dir, f), 'L')
                    image = io.imread(os.path.join(data_dir, f))
                    # reconstructed_cat_1d = np.fromstring(image.tostring(), dtype=np.uint8)
                    # reconstructed_cat_img = reconstructed_cat_1d.reshape(image.shape)
                    # print(image.shape)
                    # assert np.allclose(image, reconstructed_cat_img)

                    # print('image name:' + str(f))
                    # print('label name:' + str('label' + number + '.png'))
                    label = io.imread(os.path.join(data_dir, 'label' + number + '.png'))
                    # reconstructed_cat_1d = np.fromstring(label.tostring(), dtype=np.uint8)
                    # reconstructed_cat_img = reconstructed_cat_1d.reshape(label.shape)
                    # assert np.allclose(label, reconstructed_cat_img)
                    # label = scipy.misc.imread(os.path.join(FLAGS.dir, 'label' + number + '.png'), 'L')
                    # print('label shape:' + str(label.shape))
                    images.append(image)
                    labels.append(label)

    return images, labels


def convert_to(images, labels, name='train'):
    """Converts a dataset to tfrecords."""
    num_examples = len(images)

    if len(images) != len(labels):
        raise ValueError('The number of images %d does not match the number of labels %d.' %
                         (len(images), len(labels)))

    if FLAGS.validate == 'true':
        _write_file('train', images, labels, 0, int(num_examples * 0.8))
        _write_file('test', images, labels, int(num_examples * 0.8), num_examples)
    else:
        _write_file('train', images, labels, 0, num_examples)


def _write_file(name, images, labels, start, end):
    filename = os.path.join(FLAGS.des, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(start, end):
        image_raw = images[index].tostring()
        label_raw = labels[index].tostring()
        features = tf.train.Features(feature={
            'height': _int64_feature(images[index].shape[0]),
            'width': _int64_feature(images[index].shape[1]),
            'depth': _int64_feature(1),
            'label': _int64_feature(0),
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw)})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    print(str(end - start) + ' samples written!')


def main(unused_argv):
    # Get the data.
    images, labels = read_data()

    # Convert to Examples and write the result to TFRecords.
    convert_to(images, labels, FLAGS.name)
    # convert_to(data_sets.validation, 'validation')
    # convert_to(data_sets.test, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dirs',
        type=str,
        default='/tmp/data',
        help='Directory to download data files and write the converted result.'
    )
    parser.add_argument(
        '--des',
        type=str,
        default='/tmp/data',
        help='Destination directory.'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='train of test.'
    )
    parser.add_argument(
        '--validate',
        type=str,
        default='true',
        help='true of false.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
