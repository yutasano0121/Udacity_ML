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

# %matplotlib inline  # for jupyter notebook

# general libraries
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime

# sklearn and Boston dataset
import sklearn.model_selection
from sklearn.datasets import load_boston

# sagemaker libraries
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

import logging

working_dir = '/home/ec2-user/SageMaker/sagemaker-deployment/'
log_dir = working_dir + 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'boston_update.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:(name)s:%(message)s')


file_handler = logging.FileHandler(log_filename)
print('log file {} created'.format(log_filename))

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# make a local directly to store the split data in
data_dir = working_dir + 'data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)



session = sagemaker.Session()  # SageMaker session info (region etc.)
role = get_execution_role()  # currently assigned IAM role (i.e. access rights. See https://aws.amazon.com/iam/)

# Download the data
logger.info('Data loading')
boston = load_boston()
x_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
y_boston = pd.DataFrame(boston.target)
logger.info('Data loaded')

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
# store the data. Make sure header and index are both set False.
# also make sure that the first column stores target valriables
pd.concat([y_val, x_val], axis=1).to_csv(
    os.path.join(data_dir, 'validation.csv'),
    header=False, index=False
)

pd.concat([y_train, x_train], axis=1).to_csv(
    os.path.join(data_dir, 'train.csv'),
    header=False, index=False
)
logger.info('Data stored locally.')

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
logger.info('Data uploaded to S3.')

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
logger.info('Model fitting')
xgb.fit(
    {
        'train': s3_train,
        'validation': s3_val
    }
)


"""
Use a low-level API for deployment
so as to have a better control over
how the endpoint behaves.
"""

model_name = 'boston-update-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

primary_container = {
    'Image': container,
    'ModelDataUrl': xgb.model_data
}

model_info = session.sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=primary_container
)

# Create an endpoint configuration.
endpoint_config_name = 'boston-update-config-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

endpoint_config_info = session.sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'InstanceType': 'ml.m4.xlarge',
            'InitialVariantWeight': 1,
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'XGB-Model'
        }
    ]
)

# Create an endpoint using the configuration.
endpoint_name = 'boston-update-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

endpoint_info = session.sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

logger.info("Open the endpoint.")
# Use the endpoint.
response = session.sagemaker_runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=','.join(map(str, x_test.values[0]))
)

logger.info("Response: {}".format(response))

# Close the endpoint.
session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
logger.info("Endpoint closed.")


"""
Train a linear model.
"""

linear_container = get_image_uri(session.boto_region_name, 'linear-learner')

linear = sagemaker.estimator.Estimator(
    linear_container,
    role,
    train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
    sagemaker_session=session
)

linear.set_hyperparameters(
    feature_dim=13,
    predictor_type='regressor',
    mini_batch_size=200
)

linear.fit(
    {
        'train': s3_train,
        'validation': s3_val
    }
)

# Deploy the linear model.
linear_model_name = 'boston-update-linear-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

linear_primary_container = {
    'Image': linear_container,
    'ModelDataUrl': linear.model_data
}

linear_model_info = session.sagemaker_client.create_model(
    ModelName=linear_model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=linear_primary_container
)

# Create endpoint configuration.
linear_endpoint_config_name = 'boston-linear-endpoint-config-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

linear_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config_name,
    ProductionVariants=[
        {
            'InstanceType': 'ml.m4.xlarge',
            'InitialVariantWeight': 1,
            'InitialInstanceCount': 1,
            'ModelName': linear_model_name,
            'VariantName': 'Linear-Model'
        }
    ]
)

endpoint_name = 'boston-update-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

endpoint_info = session.sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

# Deploy the model.
response = session.sagemaker_runtime_client.invoke_endpoint(
    EndpointName = endpoint_name,
    ContentType = 'text/csv',
    Body = ','.join(map(str, x_test.values[0]))
)

# Close the endpoint.
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)


# Plot predicted/true values.
plt.scatter(y_test, y_pred)
plot.xlabel('true')
plot.ylabel('predicted')



# Remove locally stored files.
os.system('rm {}/*'.format(data_dir))
logger.info('Local file deleted. All done.')
