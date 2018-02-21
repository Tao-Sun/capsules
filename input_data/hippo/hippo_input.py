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

"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import skimage.io as io

import tensorflow as tf


def read_and_decode(filename_queue, image_height, image_width):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    height = tf.cast(features['height'], tf.int32) # tf.to_int64(features['height'])
    width = tf.cast(features['width'], tf.int32) #tf.to_int64(features['width'])

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.reshape(image, [height, width])
    # resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                                        target_height=IMAGE_HEIGHT,
    #                                                        target_width=IMAGE_WIDTH)
    image.set_shape([image_height, image_width])

    label = tf.decode_raw(features['label_raw'], tf.uint8)
    #label = tf.reshape(label, [IMAGE_HEIGHT, IMAGE_WIDTH])
    label = tf.reshape(label, [height, width])
    # resized_label = tf.image.resize_image_with_crop_or_pad(image=label,
    #                                                        target_height=IMAGE_HEIGHT,
    #                                                        target_width=IMAGE_WIDTH)
    label.set_shape([image_height, image_width])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(label, tf.float32) * (1. / 255)
    recons_label = features['label']

    return image, label, recons_label


def inputs(split, data_dir, batch_size, image_height, image_width, num_targets, file_start, file_end):
    """Reads input data num_epochs times.

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.

    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """

    file_num = file_end - file_start + 1
    if split == 'train':
        file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(1, int(0.8 * file_num))]
    elif split == 'test':
        file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(int(0.8 * file_num), file_end + 1)]

    with tf.name_scope('input'):
        if split == 'train':
            shuffle = True
        elif split == 'test':
            shuffle = False
        filename_queue = tf.train.string_input_producer(file_names, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label, recons_label = read_and_decode(filename_queue, image_height, image_width)

        features = {
            'images': image,
            'labels': label,
            'recons_label': recons_label,
            'recons_image': label
        }

        if split == 'train':
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            batched_features = tf.train.shuffle_batch(
                features, batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        elif split == 'test':
            batched_features = tf.train.batch(
                features, batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size)



        batched_features['height'] = image_height
        batched_features['width'] = image_width
        batched_features['depth'] = 1
        batched_features['num_targets'] = num_targets
        batched_features['num_classes'] = 1

        return batched_features

if __name__ == '__main__':
    tfrecords_filename = os.path.join(sys.argv[1], 'train.tfrecords')
    print(tfrecords_filename)

    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)
    resized_image, resized_annotation, _ = read_and_decode(filename_queue)

    images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                 batch_size=2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, anno = sess.run([images, annotations])

        # Let's read off 3 batches just for example
        for i in range(4):
            img, anno = sess.run([images, annotations])
            print(anno.shape)

            print('current batch')

            # We selected the batch size of two
            # So we should get two image pairs in each batch
            # Let's make sure it is random

            io.imshow(img[0, :, :], cmap='gray')
            io.show()

            io.imshow(anno[0, :, :], cmap='gray')
            io.show()

            io.imshow(img[1, :, :], cmap='gray')
            io.show()

            io.imshow(anno[1, :, :], cmap='gray')
            io.show()

        coord.request_stop()
        coord.join(threads)