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
import datetime
from googleapiclient import discovery


def batch_prediction(project_id, model_dir, test_data, output):
  api = discovery.build('ml', 'v1')
  job_id = 'deepsea_prediction_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')
  job_request = {
      'job_id': job_id,
      'prediction_input': { 
          'uri': model_dir,
          'data_format': 'TF_RECORD_GZIP',
          'input_paths': test_data,
          'output_path': output,
          'runtime_version': '1.4',
          'accelerator': {'count':1, 'type':'NVIDIA_TESLA_P100'},
          'region': 'us-west1'
      }
  }

  request = api.projects().jobs().create(body=job_request, parent='projects/' + project_id)
  request.execute()


parser = argparse.ArgumentParser()
parser.add_argument('--project',
                    required=True,
                    type=str,
                    help='Google Cloud project id.')
parser.add_argument('--model',
                    required=True,
                    type=str,
                    help='Google Storage path for the model.')
parser.add_argument('--testdata',
                    required=True,
                    type=str,
                    help='Google Storage path for test tfrecord.gz file.')
parser.add_argument('--output',
                    required=True,
                    type=str,
                    help='Google Storage path for output results.')
args, unknown = parser.parse_known_args()


batch_prediction(args.project, args.model, args.testdata, args.output)

