# Step 3: Run prediction with test data using CloudML Engine service.

## Install googleapiclient

pip install --upgrade google-api-python-client


## Get model path

Run the following to browse model path.

gsutil list [model_path]

model_path is the training output path in step 2. For example, 

gsutil list  gs://bradley-playground/deepsea/modeljob

The model path used for prediction is a sub directory under it. Something like:

gs://bradley-playground/deepsea/modeljob/export/Servo/1520037621/

Under the model path you should see at least "saved_model.pb" file and "variables" directory.


## Run prediction


In Step 1 the test data has been splitted into 2 halves. The second half is the reversed
sequences of first half. We will use both to gather predicted probabilities and then
average them as final probabilities.

Note that in batchpredict.py it is hard coded to use one P100 for the job.

For example:

python batchpredict.py --model gs://bradley-playground/deepsea/modeljob/export/Servo/1520037621/ --testdata gs://bradley-playground/deepsea/test1.tfrecord.gz --project bradley-playground --output gs://bradley-playground/deepsea/results1

python batchpredict.py --model gs://bradley-playground/deepsea/modeljob/export/Servo/1520037621/ --testdata gs://bradley-playground/deepsea/test2.tfrecord.gz --project bradley-playground --output gs://bradley-playground/deepsea/results2


## Monitor progress

Just like step 2 training, you can access prediction jobs and logs from cloud console.


## Check Results

Once the job is done, you can browse the output directory:

For example:

gsutil list -lh gs://bradley-playground/deepsea/results1

You should see "prediction.results-00000-of-00001". This JSON file holds prediction 919
probabilities and their labels for each instance. Once you get the two results ready,
you can move to next step to run a script to compute AUC, etc.

