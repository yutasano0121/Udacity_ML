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

from sklearn.preprocessing import MinMaxScaler


# set a working directory
working_dir = '/home/ec2-user/SageMaker/populationSegmentation'

# whether or not a new model is trained.
train_new = False
# the name of a trained model to be imported.
trained_job_name = 'sagemaker-pytorch-2020-05-20-20-19-07-567'

# set a data directory
data_dir = working_dir + 'data/pytorch/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# set a logger
log_dir = working_dir + 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'excercise1.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:(name)s:%(message)s\n')

file_handler = logging.FileHandler(log_filename)
print('log file {} created'.format(log_filename))

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# Set a SageMaker session.
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/excercise1'
role = sagemaker.get_execution_role()


s3 = boto3.client('s3')
# bucket name provided from Udacity
bucket_udacity = 'aws-ml-blog-sagemaker-census-segmentation'
s3_object_dict = s3.list_objects(Bucket=bucket_udacity)
files = [content['Key'] for content in s3_object_dict['Contents']]

# Fetch the first file.
fname = files[0]
print(fname)

data_obj = s3.get_object(
    Bucket=bucket_udacity,
    Key=fname
)
display(data_obj)

data_body = data_obj['Body'].read()
print("Data type: {}".format(type(data_body)))

# Read the bytes data and make it a dataframe.
data_stream = io.BytesIO(data_body)
df = pd.read_csv(data_stream, header=0, delimiter=',')
print(df.head())

df.shape
df.describe()
df.isnull().sum()  # Count NA per column.
df_clean = df.dropna()

df_clean.index = df[['State', 'County']].agg('-'.join, axis=1)
df_clean.drop(['State', 'County', 'CensusId'], axis=1, inplace=True)

df_scaled = df.apply(MinMaxScaler, axis=1)
