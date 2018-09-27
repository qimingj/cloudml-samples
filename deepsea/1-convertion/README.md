# Step 1: Convert training files from mat to tf.example


Copy train|test|valid.mat files to a local directory. Run the following:

python convert.py --input [inputdir] --output [outputdir]

The training data can be found at:

* gs://bradley-playground/deepsea/train.mat
* gs://bradley-playground/deepsea/valid.mat
* gs://bradley-playground/deepsea/test.mat

Save them to a local directory and pass the dir path as input.








