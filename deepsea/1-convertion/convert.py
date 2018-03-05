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
import h5py
import os
import scipy.io as io
import tensorflow as tf


def convert(input_dir, output_dir):
  """Convert data from mat to tf.example
  Args:
    input_dir: Local directory that should contain train.mat, valid.mat and test.mat files.
    output_dir: It will produce train.tfrecord.gz, valid.tfrecord.gz, test.tfrecord.gz files
                in the dir.
  """

  write_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

  # Process valid data which can be opened with scipy.io
  print('processing valid file.')
  input_file = os.path.join(input_dir, 'valid.mat')
  output_file = os.path.join(output_dir, 'valid.tfrecord.gz')
  data = io.loadmat(input_file)
  with tf.python_io.TFRecordWriter(output_file, write_options) as writer:
    for i in range(data['validdata'].shape[0]):
      ex_dict = {}
      ex_dict['features'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=data['validxdata'][i,:,:].transpose().reshape([-1])))
      ex_dict['labels'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=data['validdata'][i]))
      ex = tf.train.Example(features=tf.train.Features(feature=ex_dict))
      writer.write(ex.SerializeToString())

  # Process test data which can be opened with scipy.io.
  # We split test data into first half and second half because second half is the reversed
  # version of first half, and we want to run them separately and then average the prob.
  print('processing test file.')
  input_file = os.path.join(input_dir, 'test.mat')
  output_file1 = os.path.join(output_dir, 'test1.tfrecord.gz')
  output_file2 = os.path.join(output_dir, 'test2.tfrecord.gz')
  data = io.loadmat(input_file)
  with tf.python_io.TFRecordWriter(output_file1, write_options) as writer1, \
       tf.python_io.TFRecordWriter(output_file2, write_options) as writer2:
    num_instances = data['testdata'].shape[0]
    for i in range(num_instances):
      if i % 10000 == 0:
        print('processing test %d records' % i)
      ex_dict = {}
      ex_dict['features'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=data['testxdata'][i,:,:].transpose().reshape([-1])))
      ex_dict['labels'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=data['testdata'][i]))
      ex = tf.train.Example(features=tf.train.Features(feature=ex_dict))
      if i < num_instances / 2:
        writer1.write(ex.SerializeToString())
      else:
        writer2.write(ex.SerializeToString())


  # Process training data which can be opened with h5py
  print('processing train file.')
  input_file = os.path.join(input_dir, 'train.mat')
  output_file = os.path.join(output_dir, 'train.tfrecord.gz')
  with h5py.File(input_file, 'r') as f:
    with tf.python_io.TFRecordWriter(output_file, options=write_options) as writer:
      for i in range(f['traindata'].shape[-1]):
        if i % 10000 == 0:
          print('processing train %d records' % i)
        ex_dict = {}
        ex_dict['features'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=f['trainxdata'][:,:,i].reshape(-1)))
        ex_dict['labels'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=f['traindata'][:, i]))
        ex = tf.train.Example(features=tf.train.Features(feature=ex_dict))
        writer.write(ex.SerializeToString())


parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    required=True,
                    type=str,
                    help='Local directory which contains train|test|valid.mat files.')
parser.add_argument('--output',
                    required=True,
                    type=str,
                    help='Local output directory.')
args, unknown = parser.parse_known_args()


convert(args.input, args.output)
