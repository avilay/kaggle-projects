# kaggle-projects

## Use Cases

### Download Kaggle data

1. SSH into the bastion in AWS/us-east-1.
2. Run the `kaggle` cli to download data to the local SSD.
3. Run the `aws s3 cp` cli to upload the data to `s3://avilabs-mldata/<project-name>`.

### Sample a local copy

1. Go to working area -
    * Start a Terminal in AWS Sagemaker Notebook.
    * Go to local terminal.
2. Run the program to download a sample in the range (0, 1] to the local SSD.

### Run Experiment

1. Instantiate a `Dataset` class and provide a data source. This can be `file:///` or `s3://` 
2. Additionally provide cache details - a) whether to cache the data, b) where to cache it, this can be either Redis or local memory.
3. Run the training. The training metrics are written to `file:///`.  Should this be moved to S3 in the background?

### Offline Pre-processing

1. Read the data from the source. This can be `file:///` or `s3://`.
2. If needed load the data in a cache. This can be either Redis or local memory.
3. Pre-process each example and write it back to `file:///` or `s3://`. 





