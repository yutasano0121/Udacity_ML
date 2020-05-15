"""
Outline for using AWS SageMaker:
1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3 (storage).
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job). (Can be done after deployment.)
6. Deploy the trained model.
7. Use the deployed model.
"""

%matplotlib inline  # for jupyter notebook

# general libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn and Boston dataset
import sklearn.model_selection
from sklearn.datasets import load_boston

# sagemaker libraries
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_url
from sagemaker.predictor import csv_serializer


# Create logger
working_dir = '~/SageMaker/sagemaker-deployment/'
log_dir = working_dir + 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = log_dir + 'boston_tutorial.log'
import(logger)


# make a local directly to store the split data in
data_dir = working_dir + 'data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


log_filename =

session = sagemaker.Session()  # SageMaker session info (region etc.)
role = get_execution_role()  # currently assigned IAM role (i.e. access rights. See https://aws.amazon.com/iam/)

# Download the data
boston = load_boston
x_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
y_boston = pd.DataFrame(boston.target)

# split the data into train and test datasets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x_boston,
    y_boston,
    test_size=0.33
)

# split train into train and validation sets
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    x_train,
    y_train,
    test_size=0.33
)


"""
To run SageMaker, data need to be in a S3 storage bucket.
"""

# make a local directly to store the split data in
working_dir = '/home/ec2-user/SageMaker/sagemaker-deployment/'
data_dir = working_dir + 'data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# store the data. Make sure header and index are both set False.
# also make sure that the first column stores target valriables
pd.concat([y_val, x_val], axis=1).to_csv(
    os.path.join(data_dir, 'validation_boston.csv', header=False, index=False)
)

pd.concat([y_train, x_train], axis=1).to_csv(
    os.path.join(data_dir, 'train_boston.csv', header=False, index=False)
)

# Upload the data to S3.
prefix = 'boston_xgboost_high'  # S3 folder name

val_location = session.upload_data(
    os.path.join(data_dir, 'validation.csv'),
    key_prefix=prefix
)

train_location = session.upload_data(
    os.path.join(data_dir, 'train.csv'),
    key_prefix=prefix
)


"""
Train XGBoost using a container provided by SageMaker.
"""

# Define a container.
container = get_image_uri(session.boto_region_name, 'xgboost')

# Construct a XGBoost extimator.
xgb = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
    sagemaker_session=session
)

# Set hyperparameters.
xgb.set_hyperparameters(
    max_depth=5,  # default = 5. Lower values regularize overfitting.
    eta=0.2,  # learning rate. default = 0.3.
    gamma=4,  # the minimum loss required to make a split. default = 0.
    min_child_weight=6,  # minimum sum of weights for all observations in a child.
    # default = 1. Higher values regularize overfitting.
    subsample=0.8,  # fraction of samples randomely sampled for each tree. default = 1.
    objective='reg:linear',
    early_stopping_rounds=10,
    num_round=200
)

# Specify the data location in S3.
s3_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_val = sagemaker.s3_input(s3_data=val_location, content_type='csv')

# Fit the model.
xgb.fit({
    'train': s3_train,
    'validation': s3_val
})


"""
Deploy the trained model to make an estimate.
Do not forget to shut down a deployed model after run
by "DEPLOYED_MODEL.delete_endpoint."
"""

# Create an endpoint.
xgb_predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)


# Specify the format of input data.
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer


# make a prediction
y_pred = xgb_predictor.predict(x_test.values).decode('utf-8')  # Returns a string
y_pred = np.fromstring(y_pred, sep=',')  # Make it to np.array.


# Plot predicted/true values.
plt.scatter(y_test, y_pred)
plot.xlabel('true')
plot.ylabel('predicted')


# Close the endpoint.
xgb_predictor.delete_endpoint()


# Remove locally stored files.
!rm $data_dir/*
