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

from sklearn.model_selection import train_test_split

from evaluate import evaluate, print_fraudRatio


# set a working directory
working_dir = '/home/ec2-user/SageMaker/fraudDetection/'

# whether or not a new model is trained.
train_new = False
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
log_filename = log_dir + 'excercise2.log'

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
prefix = 'sagemaker/creditcard'
role = sagemaker.get_execution_role()
s3 = boto3.client('s3')


# Download data.
local_data = os.path.join(data_dir, 'creditcard.csv')
if not os.path.exists(local_data):
    logger.info('Data downloaded.')
    subprocess.check_call(
        'wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c534768_creditcardfraud/creditcardfraud.zip \
        -P {}'.format(data_dir),
        shell=True
    )
    subprocess.check_call(
        'unzip {} -d {}'.format(
            os.path.join(data_dir, 'creditcardfraud.zip'),
            data_dir
        ), shell=True
    )

df = pd.read_csv(local_data)
print('Data shape (rows, cols): ', df.shape)
print(df.head())

# Print a ratio of fraudrant transactions.
print_fraudRatio(df)

# Split train and test.
y = df.Class
x = df.drop('Class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)
print(
    "Fraudulent ratio in train: {}".format(
        y_train.sum() / len(y_train)
    )
)

# Store the data locally
x_test.to_csv(
    os.path.join(data_dir, 'x_test.csv'),
    header=False, index=False
)
y_test.to_csv(
    os.path.join(data_dir, 'y_test.csv'),
    header=False, index=False
)

pd.concat([y_train, x_train], axis=1).to_csv(
    os.path.join(data_dir, 'train.csv'),
    header=False, index=False
)
logger.info('Data stored locally.')

test_location = session.upload_data(
    os.path.join(data_dir, 'x_test.csv'),
    key_prefix=prefix
)



# Start linear learning.
output_dir = 's3://{}/{}'.format(bucket, prefix)
ll = LinearLearner(
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    predictor_type='binary_classifier',
    output_path=output_dir,
    sagemaker_session=session,
    epochs=15
)

# Make test x and y numpy arrays.
# Default dype is 'float 64' but 'float32' is required for both features and labels.
x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
train_formatted = ll.record_set(x_train, labels=y_train)

logger.info('Linear Learner started.')
ll.fit(train_formatted)
logger.info('Linear Learner done.')

# Make test data numpy arrays.
x_test = np.array(x_test).astype('float32')
y_test = np.array(y_test).astype('float32')

ll_predictor = ll.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)


# get metrics for linear predictor
metrics = evaluate(
    ll_predictor,
    x_test,
    y_test,
    verbose=True
)

ll_predictor.delete_endpoint()


"""
Create a LinearLearner with a higher precision.
"""

ll_recall = LinearLearner(
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    predictor_type='binary_classifier',
    output_path=output_dir,
    sagemaker_session=session,
    epochs=15,
    binary_classifier_model_selection_criteria='precision_at_target_recall',
    target_recall=0.9
)

logger.info('Linear Learner (high recall) started.')
ll_recall.fit(train_formatted)
logger.info('Linear Learner (high recall) done.')

ll_recall_predictor = ll_recall.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

metrics_recall = evaluate(
    ll_recall_predictor,
    x_test,
    y_test,
    verbose=True
)

ll_recall_predictor.delete_endpoint()

"""
Create a LinearLearner with a higher precision and balanced input.
"""

ll_balanced = LinearLearner(
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    predictor_type='binary_classifier',
    output_path=output_dir,
    sagemaker_session=session,
    epochs=15,
    binary_classifier_model_selection_criteria='precision_at_target_recall',
    target_recall=0.9,
    positive_example_weight_mult='balanced'
)

logger.info('Linear Learner (balanced) started.')
ll_balanced.fit(train_formatted)
logger.info('Linear Learner (balanced) done.')

ll_balanced_predictor = ll_balanced.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

metrics_balanced = evaluate(
    ll_balanced_predictor,
    x_test,
    y_test,
    verbose=True
)

ll_balanced_predictor.delete_endpoint()
