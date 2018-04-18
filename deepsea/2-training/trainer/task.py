# Copyright 2018 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.


import os

# os.environ["GRPC_VERBOSITY"] = "debug"
# os.environ["GRPC_TRACE"] = "call_error,client_channel,channel"


import argparse
import json
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


BATCH_SIZE = 128
OUTPUTS = 919


def parse_example(serialized):
  parsed = tf.parse_single_example(
      serialized,
      features={
        'features': tf.FixedLenFeature([4000], dtype=tf.int64),
        'labels': tf.FixedLenFeature([919], dtype=tf.int64),
      })
  parsed['features'] = tf.cast(tf.reshape(parsed['features'], [1000, 4]), tf.float32)
  return parsed


def create_input_fn(filename, mode=tf.estimator.ModeKeys.TRAIN):  
  """Creates an input_fn for estimator in training or evaluation."""

  def _input_fn():
    """Returns named features and labels, as required by Estimator."""  
    # could be a path to one file or a file pattern.
    dataset = tf.data.TFRecordDataset([filename], compression_type='GZIP')
    dataset = dataset.map(parse_example)
    dataset = dataset.map(lambda input_batch: (input_batch['features'], input_batch['labels']))
    if mode==tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

  return _input_fn


def model_fn(features, labels, mode, params):
  """Define the inference model."""
  
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  if mode == tf.estimator.ModeKeys.PREDICT:
    labels = features['labels']
    features = features['features']

  net = tf.cast(features, tf.float32)
  # net = tf.layers.batch_normalization(net, training=is_training) 
  net = tf.layers.conv1d(net, filters=320, kernel_size=8)
  net = tf.layers.conv1d(net, filters=320, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=1)
  if mode == tf.estimator.ModeKeys.TRAIN:
    net = tf.nn.dropout(net, 0.8)

  # net = tf.layers.batch_normalization(net, training=is_training) 
  net = tf.layers.conv1d(net, filters=320, kernel_size=8)
  net = tf.layers.conv1d(net, filters=320, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=2)
  if mode == tf.estimator.ModeKeys.TRAIN:
    net = tf.nn.dropout(net, 0.8)

  # net = tf.layers.batch_normalization(net, training=is_training) 
  net = tf.layers.conv1d(net, filters=480, kernel_size=8)
  net = tf.layers.conv1d(net, filters=480, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=2)
  if mode == tf.estimator.ModeKeys.TRAIN:
    net = tf.nn.dropout(net, 0.8)

  # net = tf.layers.batch_normalization(net, training=is_training)
  net = tf.layers.conv1d(net, filters=640, kernel_size=8)
  net = tf.layers.conv1d(net, filters=640, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=2)
  if mode == tf.estimator.ModeKeys.TRAIN:
    net = tf.nn.dropout(net, 0.8)

  # net = tf.layers.batch_normalization(net, training=is_training) 
  net = tf.layers.conv1d(net, filters=960, kernel_size=8)
  net = tf.layers.conv1d(net, filters=960, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=2)
  if mode == tf.estimator.ModeKeys.TRAIN:
    net = tf.nn.dropout(net, 0.8)

  # net = tf.layers.batch_normalization(net, training=is_training) 
  net = tf.layers.conv1d(net, filters=960, kernel_size=8)
  net = tf.layers.conv1d(net, filters=960, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=2)  
  if mode == tf.estimator.ModeKeys.TRAIN:
    net = tf.nn.dropout(net, 0.8)

  # net = tf.layers.batch_normalization(net, training=is_training) 
  net = tf.reshape(net, [tf.shape(net)[0], 14400])
  logits = tf.layers.dense(net, OUTPUTS)

  probs = tf.sigmoid(logits)
  predictions = tf.cast(tf.round(probs), tf.int64)

  # predictions are all we need when mode is not train/eval. 
  predictions_dict = {"predicted": probs}
  if mode == tf.estimator.ModeKeys.PREDICT:
    # Uncomment the following if you are doing predictions with no truth passed in.
    # predictions_dict['features'] = features['features']
    predictions_dict['labels'] = labels
  
  # If train/evaluation, we'll need to compute loss.
  # If train, we will also need to create an optimizer.
  loss, train_op, eval_metric_ops = None, None, None
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    if mode == tf.estimator.ModeKeys.EVAL:
      auc = tf.metrics.auc(labels, probs)
      eval_metric_ops = {
          "auc": auc,
      }
    if mode == tf.estimator.ModeKeys.TRAIN:
    
      def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(
                learning_rate=0.1,
                global_step=tf.train.get_global_step(),
                decay_steps=700000,
                decay_rate=0.3,
                staircase=True)

      optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
      train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

  export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions_dict)
  }

  # return ModelFnOps as Estimator requires.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions_dict,
      export_outputs=export_outputs,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def get_train(train_file):
  return create_input_fn(train_file, mode=tf.estimator.ModeKeys.TRAIN)


def get_eval(eval_file):
  return create_input_fn(eval_file, mode=tf.estimator.ModeKeys.EVAL)


def serving_input_fn():
  input_placeholder = tf.placeholder(tf.string, [None])
  feature_placeholders = {
      'features': input_placeholder
  }
  features = tf.parse_example(
      input_placeholder,
      features={
        'features': tf.FixedLenFeature([4000], dtype=tf.int64),
        'labels': tf.FixedLenFeature([919], dtype=tf.int64),
      })
  features['features'] = tf.cast(tf.reshape(features['features'], [-1, 1000, 4]), tf.float32)
  return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    type=str,
                    help='Training files local or GCS')
parser.add_argument('--eval',
                    required=True,
                    type=str,
                    help='Evaluation files local or GCS')
parser.add_argument('--job-dir',
                    required=True,
                    type=str,
                    help='GCS or local dir to write checkpoints and export model')
parser.add_argument('--train-steps',
                    type=int)
parser.add_argument('--evalonly',
                    action='store_true',
                    help='Whether to run evaluation only on eval set')

args, unknown = parser.parse_known_args()

distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=args.job_dir,
                                   config=tf.estimator.RunConfig(save_checkpoints_steps=1000, train_distribute=distribution))

if args.evalonly:
  estimator.evaluate(input_fn=get_eval(args.eval))
else:
  exporter = tf.estimator.FinalExporter('export', serving_input_fn)
  train_spec = tf.estimator.TrainSpec(input_fn=get_train(args.train), max_steps=args.train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=get_eval(args.eval), exporters=[exporter], steps=None)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
