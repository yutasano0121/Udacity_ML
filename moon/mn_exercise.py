"""
Activate 'mxnet_p36' environment!
"""

import pandas as pd
import numpy as np
import os
import subprocess
import io
import logging

import matplotlib.pyplot as plt
import matplotlib

import boto3
import sagemaker
from sagemaker import LinearLearner
from sagemaker.pytorch import PyTorch, PyTorchModel

from sklearn.model_selection import train_test_split

from func import check_data_in_S3, evaluate


# set a working directory
working_dir = '/home/ec2-user/SageMaker/moon/'

# whether or not a new model is trained.
train_new = True
# the name of a trained model to be imported.
trained_job_name = 'sagemaker-pytorch-2020-05-20-20-19-07-567'

# set a data directory
data_dir = os.path.join(working_dir, 'data/')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# set a logger
log_dir = os.path.join(working_dir, 'log/')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'mn_excercise.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:(name)s:%(message)s\n'
)

file_handler = logging.FileHandler(log_filename)
print('log file {} created'.format(log_filename))

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# Set a SageMaker session.
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/moon'
role = sagemaker.get_execution_role()
s3 = boto3.client('s3')


# Generate moon data.
np.random.seed(0)
x, y = make_moons(600, noise=0.25)  # x = 2D points, y = binary class labels
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

logger.info(
    "Data split into train and test.\n \
    Train shape: {}\n \
    Test shape: {}\n".format(
        x_train.shape,
        x_test.shape
    )
)

# Save the train and test data in the local.
pd.concat([pd.DataFrame(y_train), pd.DataFrame(x_train)]).to_csv(
    os.path.join(data_dir, 'train.csv'),
    header=False,
    index=False
)
pd.DataFrame(y_test).to_csv(
    os.path.join(data_dir, 'y_test.csv'),
    header=False,
    index=False
)
pd.DataFrame(x_test).to_csv(
    os.path.join(data_dir, 'x_test.csv'),
    header=False,
    index=False
)

# Upload the data to S3.
check_data_in_S3(data_dir, bucket, prefix, s3)

# Instantiate a PyTorch model.
source_dir = os.path.join(working_dir, 'Udacity_ML/moon/source/')
model = PyTorch(
    entry_point='train.py',
    source_dir=source_dir,
    train_instance_typr='ml.c4.xlarge',
    role=role,
    sagemaker_session=session,
    framework_version='1.59',  # latest version of PyTorch
    train_instance_count=1,
    output_path='s3://{}/{}'.format(bucket, prefix),
    hyperparameters={  # Parameters specified in train.py
        'input_dim': 2,  # num of features
        'hidden_dim': 20,
        'output_dim': 1,
        'epochs': 80  # could change to higher
    }
)

# Train the model or attach a pre-trained one.
if train_new:
    logger.info("Train the model.")
    model.fit({'train': input_data})
else:
    logger.info("Load a pre-trained model.")
    estimator = model.attach(trained_job_name)

# Create a estimator.
estimator = PyTorchModel(
    model_data=model.model_data,
    entry_point='predict.py',
    source_dir=source_dir,
    role=role,
    sagemaker_session=session,
    framework_version='1.59'
)

# Deploy the trained model.
print("Endpoint created.")
estimator_endpoint = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Evaluate a performance of the model.
metrics = evaluate(estimator_endpoint, x_test, y_test, True)

# Delete the endpoint.
estimator_endpoint.delete_endpoint()
print("Endpoint deleted.")
