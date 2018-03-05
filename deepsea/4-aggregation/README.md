# Step 4: Aggregation -- an example to use prediction results.


The example demonstrate how to aggregate the prediction results and calculate AUC. It takes
two prediction results, one contains reversed sequence of the other but with the same labels.

Run the following:

python aggregate.py --input1 [gcs or local path] --input2 [gcs or local path] --output [local_dir]


For example: 

python aggregate.py --input1 gs://bradley-playground/deepsea/results1/prediction.results-00000-of-00001 --input2 gs://bradley-playground/deepsea/results2/prediction.results-00000-of-00001 --output ./


