# Step 2: Convert training files from mat to tf.example


## Install GCloud

https://cloud.google.com/sdk/downloads


## Gcloud config

1. ```gcloud auth login``` and finish the auth process.
2. Run ```gcloud config set project [your-project-id]``` to set default project.


## Copy Converted training files to storage

If you don't have a storage bucket yet, create one first. For example, run:

```gsutil mb gs://bradley-playground-deepsea```

Then copy the converted tfrecord files to storage so that training can access them.

For example, run:

```gsutil cp ./*.tfrecord.gz gs://bradley-playground/deepsea/data```


## Enable Machine learning API in your cloud project.


Go to https://console.cloud.google.com, select your project. Click API & Services in the left menu,
click Dashboard Click “Enable APIs and Services” in the upper menu bar. Search for “Machine Learning”,
and find “Google Cloud Machine Learning Engine”. Enable it.


## Run Training

gcloud ml-engine jobs submit training [jobname] --runtime-version=1.6 --module-name trainer.task --package-path=./trainer --job-dir=[gs://your-output-path] --config config.yaml --region us-central1 -- --train=[gs://your-train-tfrecord-file] --eval=[gs://your-valid-tfrecord-file] --train-steps=500000

For example:

gcloud ml-engine jobs submit training deepseajob --runtime-version=1.6 --module-name trainer.task --package-path=./trainer --job-dir=gs://bradley-playground/deepsea/modeljob --config config.yaml --region us-central1 -- --train=gs://bradley-playground/deepsea/data/train.tfrecord.gz --eval=gs://bradley-playground/deepsea/data/valid.tfrecord.gz --train-steps=500000

gcloud ml-engine jobs submit training deepseaeval --runtime-version=1.6 --module-name trainer.task --package-path=./trainer --job-dir=gs://bradley-playground/deepsea/modeljob8 --config config.yaml --region us-central1 -- --eval=gs://bradley-playground/test1.tfrecord.gz --evalonly

Note that in config.yaml P100 single node training is specified.


## Monitor progress

1. You can go to cloud console to list jobs and view logs for each. For example, https://console.cloud.google.com/mlengine/jobs.
2. You can also start tensorboard to see loss, AUC curves, etc. For example, run:

   ```tensorboard --logdir gs://bradley-playground/deepsea/modeljob```

   then open browser to watch the curves.
