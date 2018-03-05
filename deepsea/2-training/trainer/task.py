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


import argparse
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


BATCH_SIZE = 64
OUTPUTS = 919


def parse_example(serialized):
  parsed = tf.parse_single_example(
      serialized,
      features={
        'features': tf.FixedLenFeature([4000], dtype=tf.int64),
        'labels': tf.FixedLenFeature([919], dtype=tf.int64),
      })
  parsed['features'] = tf.reshape(parsed['features'], [1000, 4])
  return parsed


def create_input_fn(filename, mode=tf.contrib.learn.ModeKeys.TRAIN):  
  """Creates an input_fn for estimator in training or evaluation."""

  def _input_fn():
    """Returns named features and labels, as required by Estimator."""  
    # could be a path to one file or a file pattern.
    dataset = tf.data.TFRecordDataset([filename], compression_type='GZIP')
    dataset = dataset.map(parse_example)
    if mode==tf.contrib.learn.ModeKeys.TRAIN:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    input_batch = iterator.get_next()
    features = tf.cast(input_batch['features'], tf.float32)
    return {'features': features}, input_batch['labels']   # dict of features, target

  return _input_fn


def model_fn(features_dict, targets, mode):
  """Define the inference model."""

  net = features_dict['features']
  
  net = tf.layers.conv1d(net, filters=320, kernel_size=8)
  net = tf.layers.conv1d(net, filters=320, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=2)
  # if mode == tf.contrib.learn.ModeKeys.TRAIN:
  #   net = tf.nn.dropout(net, 0.1)
  net = tf.layers.conv1d(net, filters=480, kernel_size=8)
  net = tf.layers.conv1d(net, filters=480, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=4)
  # if mode == tf.contrib.learn.ModeKeys.TRAIN:   
  #   net = tf.nn.dropout(net, 0.1)  
  net = tf.layers.conv1d(net, filters=960, kernel_size=8)
  net = tf.layers.conv1d(net, filters=960, kernel_size=8)
  net = tf.layers.max_pooling1d(net, 4, strides=4)
  # if mode == tf.contrib.learn.ModeKeys.TRAIN:   
  #   net = tf.nn.dropout(net, 0.1)

  net = tf.reshape(net, [tf.shape(net)[0], 24960])
  net = tf.layers.dense(net, OUTPUTS * 2)
  logits = tf.layers.dense(net, OUTPUTS)

  probs = tf.sigmoid(logits)
  predictions = tf.cast(tf.round(probs), tf.int64)

  # predictions are all we need when mode is not train/eval. 
  predictions_dict = {"predicted": probs}
  if mode == tf.contrib.learn.ModeKeys.INFER:
    # Uncomment the following if you are doing predictions with no truth passed in.
    # predictions_dict['features'] = features_dict['features']
    predictions_dict['labels'] = features_dict['labels']
  
  # If train/evaluation, we'll need to compute loss.
  # If train, we will also need to create an optimizer.
  loss, train_op, eval_metric_ops = None, None, None
  if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
    loss = tf.losses.sigmoid_cross_entropy(targets, logits)
    auc = tf.metrics.auc(targets, probs)
    eval_metric_ops = {
      "auc": auc,
    }
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
    
      def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(
                learning_rate=0.1,
                global_step=tf.contrib.framework.get_global_step(),
                decay_steps=400000,
                decay_rate=0.3,
                staircase=True)

      train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=0.1,
          optimizer='SGD',
          clip_gradients=5.0,
          learning_rate_decay_fn=_learning_rate_decay_fn)

  # return ModelFnOps as Estimator requires.
  return tflearn.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def get_train(train_file):
  return create_input_fn(train_file, mode=tf.contrib.learn.ModeKeys.TRAIN)


def get_eval(eval_file):
  return create_input_fn(eval_file, mode=tf.contrib.learn.ModeKeys.EVAL)


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
  
  return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders
  )


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


def experiment_fn(output_dir):
    """An experiment_fn required for Estimator API to run training."""

    estimator = tflearn.Estimator(model_fn=model_fn,
                                  model_dir=args.job_dir,
                                  config=tf.contrib.learn.RunConfig(save_checkpoints_steps=1000))
    return tflearn.Experiment(
        estimator,
        train_input_fn=get_train(args.train),
        eval_input_fn=get_eval(args.eval),
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        train_steps=args.train_steps,
        eval_steps=None,
    )


if args.evalonly:
  estimator = tflearn.Estimator(model_fn=model_fn,
                                model_dir=args.job_dir,
                                config=tf.contrib.learn.RunConfig(save_checkpoints_steps=1000))
  estimator.evaluate(input_fn=get_eval(args.eval))
else:
  learn_runner.run(experiment_fn, args.job_dir)
