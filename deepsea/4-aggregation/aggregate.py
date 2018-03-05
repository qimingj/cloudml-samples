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
from itertools import izip
import json
import numpy as np
import os
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.lib.io import file_io


OUTPUTS = 919

def aggregate(input1, input2, output):
  
  output_file = os.path.join(output, 'auc.txt')
  with file_io.FileIO(input1, 'r') as f1, file_io.FileIO(input2, 'r') as f2:
    probs = []
    labels = []
    i = 0
    for line1, line2 in izip(f1, f2):
      if i % 10000 == 0:
        print('processing %d records' % i)
      r1 = json.loads(line1)
      r2 = json.loads(line2)
      assert(r1['labels'] == r2['labels'])
      labels.append(r1['labels'])
      probs.append((np.array(r1['predicted']) + np.array(r2['predicted'])) / 2)
      i += 1

  probs = np.array(probs)
  labels = np.array(labels)
  scores = []
  for i in range(OUTPUTS):
    if all(x == 0 for x in labels[:, i]):
      print('output %d all 0' % i)
    else:
      scores.append(metrics.roc_auc_score(labels[:, i], probs[:, i], average=None))
    
  with file_io.FileIO(output_file, 'w') as fw:
    for score in scores:
      fw.write('%f\n' % score)
    fw.write('average: %f' % (sum(scores) / float(len(scores))))
    

parser = argparse.ArgumentParser()
parser.add_argument('--input1',
                    required=True,
                    type=str,
                    help='the first prediction result file.')
parser.add_argument('--input2',
                    required=True,
                    type=str,
                    help='the second prediction result file.')
parser.add_argument('--output',
                    required=True,
                    type=str,
                    help='the output directory.')
args, unknown = parser.parse_known_args()


aggregate(args.input1, args.input2, args.output)
