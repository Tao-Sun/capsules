# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Capsule Model class. Adds the inference ops by attaching capsule layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model
from models.layers import layers
from models.layers import variables

import numpy as np


class CapsuleModel(model.Model):
  """A multi GPU Model with capsule layers.

  The inference graph includes a 256-Channel ReLU convolution layer,
  a 32x8-Squash convolution layer and a 10x8 Squash with routing Capsule
  layer.
  """

  def _build_capsule(self, input_tensor, num_classes):
    """Adds the capsule layers.

    A slim convolutional capsule layer transforms the input tensor to capsule
    format. The nonlinearity for slim convolutional capsule is squash function
    but there is no routing and each spatial instantiation of capsule is derived
    as traditional convolutional layer.
    In order to connect the convolutional capsule layer to the top fully
    connected capsule layer the grid position of convolution capsule is
    merged with different types of capsules dimmension and capsule2 learns
    different transformations for each of them.

    Args:
      input_tensor: 5D input tensor, shape: [batch, 1, 256, height, width].
      num_classes: Number of object categories. Used as the output dimmension.
    Returns:
      A 3D tensor of the top capsule layer with 10 capsule embeddings.
    """
    capsule1 = layers.conv_slim_capsule(
        input_tensor,
        input_dim=1,
        output_dim=self._hparams.num_prime_capsules,
        layer_name='conv_capsule1',
        num_routing=1,
        input_atoms=256,
        output_atoms=8,
        stride=2,
        kernel_size=9,
        padding=self._hparams.padding,
        leaky=self._hparams.leaky,)
    capsule1_atom_last = tf.transpose(capsule1, [0, 1, 3, 4, 2])
    capsule1_3d = tf.reshape(capsule1_atom_last,
                             [tf.shape(input_tensor)[0], -1, 8])
    _, _, _, height, width = capsule1.get_shape()
    input_dim = self._hparams.num_prime_capsules * height.value * width.value
    return layers.capsule(
        input_tensor=capsule1_3d,
        input_dim=input_dim,
        output_dim=num_classes,
        layer_name='capsule2',
        input_atoms=8,
        output_atoms=64,
        num_routing=self._hparams.routing,
        leaky=self._hparams.leaky,)

  def _summarize_remakes(self, features, remakes):
    """Adds an image summary consisting original, target and remake images.

    Reshapes all images to 3D from flattened and transposes them to be in depth
    last order. For each target concats original, target and remake image
    vertically and concats all the target columns horizantally.
    Handles up to two targets.

    Args:
      features: A dictionary of input data containing the dimmension information
        and the input images.
      remakes: A list of network reconstructions.
    """
    image_dim = features['height']
    image_depth = features['depth']

    images = []
    images.append(features['images'])
    images.append(features['recons_image'])
    images += remakes
    if features['num_targets'] == 2:
      images.append(features['spare_image'])

    images_3d = []
    for image in images:
      image_3d = tf.reshape(image, [-1, image_depth, image_dim, image_dim])
      images_3d.append(tf.transpose(image_3d, [0, 2, 3, 1]))

    image_remake = tf.concat(images_3d[:3], axis=1)

    if features['num_targets'] == 2:
      # pylint: disable=unbalanced-tuple-unpacking
      original, _, _, remake, target = images_3d
      image_remake_2 = tf.concat([original, target, remake], axis=1)
      image_remake = tf.concat([image_remake, image_remake_2], axis=2)

    tf.summary.image('reconstruction', image_remake, 10)

  def _remake(self, features, capsule_embedding):
    """Adds the reconstruction subnetwork to build the remakes.

    This subnetwork shares the variables between different target remakes. It
    adds the subnetwork for the first target and reuses the weight variables
    for the second one.

    Args:
      features: A dictionary of input data containing the dimmension information
        and the input images and labels.
      capsule_embedding: A 3D tensor of shape [batch, 10, 16] containing
        network embeddings for each digit in the image if present.
    Returns:
      A list of network remakes of the targets.
    """

    num_pixels = features['depth'] * features['height'] * features['width']
    fc_remakes = []
    targets = [(features['recons_label'], features['recons_image'])]
    if features['num_targets'] == 2:
      targets.append((features['spare_label'], features['spare_image']))
    assert(features['recons_label'].get_shape() == features['spare_label'].get_shape())


    with tf.name_scope('recons'):
      for i in range(features['num_targets']):
        label, image = targets[i]
        print('label shape:' + str(label.get_shape()))
        print('capsule_embedding shape:' + str(capsule_embedding.get_shape()))
        fc_remakes.append(
            layers.fc_recons(
                number=i,
                capsule_mask=tf.one_hot(label, features['num_classes']),
                num_atoms=64,
                capsule_embedding=capsule_embedding,
                layer_sizes=[1680],
                num_pixels=num_pixels,
                reuse=(i > 0),
                recons_image=features['recons_image'],
                num_targets=features['num_targets']))

    fc_2d_stack = tf.stack(fc_remakes, axis=1)
    fc_2d = tf.reshape(fc_2d_stack, [-1, features['num_targets'] * 80, 3, 7])
    print('fc_remakes[0] shape:' + str(fc_remakes[0].get_shape()))
    label_logits = tf.transpose(layers.deconv(fc_2d, False), perm=[0, 2, 3, 1])
    label_2d = tf.argmax(label_logits, axis=3)

    with tf.name_scope('loss'):
        image_2d = tf.cast(features['recons_image'], tf.int32)
        #image_2d = tf.contrib.layers.flatten(features['recons_image'])
        #_, recons_logits = tf.split(label_prob, [1, 1], axis=1)
        #logits_2d = tf.contrib.layers.flatten(recons_logits)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=image_2d,
                                                                       logits=label_logits)
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=image_2d, logits=logits_2d)

        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        # batch_loss = tf.reduce_mean(loss)
        balanced_loss1 = cross_entropy_loss

        balance_factor = 0.8
        capsule_logits = tf.norm(capsule_embedding, axis=-1)
        flatten_labels = tf.contrib.layers.flatten(tf.cast(label_2d, tf.float32))
        positive_percent = tf.divide(tf.reduce_sum(flatten_labels, axis=1),
                                     tf.cast(tf.shape(flatten_labels)[1], tf.float32))
        negative_percent = 1 - positive_percent
        label_percent = tf.stack([positive_percent, negative_percent], axis=1)
        balanced_loss2 = balance_factor * tf.reduce_mean(tf.pow(capsule_logits - label_percent, 2))

        tf.add_to_collection('losses', balanced_loss1)
        # tf.add_to_collection('losses', balanced_loss2)
        tf.summary.scalar('reconstruction_error', balanced_loss1)
        tf.summary.scalar('reconstruction_error', balanced_loss2)

    if self._hparams.verbose:
      self._summarize_remakes(features, fc_remakes)

    return label_2d

  def inference(self, features):
    """Adds the inference graph ops.

    Builds the architecture of the neural net to drive logits from features.
    The inference graph includes a convolution layer, a primary capsule layer
    and a 10-capsule final layer. Optionally, it also adds the reconstruction
    network on top of the 10-capsule final layer.

    Args:
      features: Dictionary of batched feature tensors like images and labels.
    Returns:
      A model.Inferred named tuple of expected outputs of the model like
      'logits' and 'recons' for the reconstructions.
    """

    image_height = features['height']
    image_width = features['width']
    image_depth = features['depth']
    image = features['images']
    image_4d = tf.reshape(image, [-1, image_depth, image_height, image_width])

    # ReLU Convolution
    with tf.variable_scope('conv1') as scope:
      kernel = variables.weight_variable(
          shape=[9, 9, image_depth, 256], stddev=5e-2,
          verbose=self._hparams.verbose)
      biases = variables.bias_variable([256], verbose=self._hparams.verbose)
      conv1 = tf.nn.conv2d(
          image_4d,
          kernel, [1, 1, 1, 1],
          padding=self._hparams.padding,
          data_format='NCHW')
      pre_activation = tf.nn.bias_add(conv1, biases, data_format='NCHW')
      relu1 = tf.nn.relu(pre_activation, name=scope.name)
      if self._hparams.verbose:
        tf.summary.histogram('activation', relu1)
    hidden1 = tf.expand_dims(relu1, 1)

    # Capsules
    capsule_output = self._build_capsule(hidden1, features['num_classes'])
    logits = tf.norm(capsule_output, axis=-1)

    # Reconstruction
    if self._hparams.remake:
      remake = self._remake(features, capsule_output)
    else:
      remake = None

    return model.Inferred(logits, remake)
