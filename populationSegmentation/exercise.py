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
import mxnet as mx


from sklearn.preprocessing import MinMaxScaler

train_from_scratch = False

# set a working directory
working_dir = '/home/ec2-user/SageMaker/populationSegmentation/'

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
prefix = 'sagemaker/counties'
role = sagemaker.get_execution_role()
s3 = boto3.client('s3')



if train_from_scratch:
    if os.path.exists(
        os.path.join(data_dir, 'df_scaled.csv')
    ):  # Load preprocessed data locally.
        df_scaled = pd.read_csv(
            os.path.join(data_dir, 'df_scaled.csv'),
            index_col=0
        )
        print("df_scaled loaded locally.")

    else:  # Load and preprocess the data from S3.
        print("Data being loaded from S3.")

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

        df_clean.index = df_clean[['State', 'County']].agg('-'.join, axis=1)
        df_clean.drop(['State', 'County', 'CensusId'], axis=1, inplace=True)

        # Scale columns for PCA
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_clean)
        df_scaled = pd.DataFrame(
            df_scaled, 
            index=df_clean.index,
            columns=df_clean.columns
        )
        df_scaled.to_csv(os.path.join(data_dir, 'df_scaled.csv'))
        print("df_scaled saved locally.")


    # Set a S3 directory to upload results to.
    s3_outdir = 's3://{}/{}/'.format(bucket, prefix)
    print("S3 directory set to {}".format(s3_outdir))

    # Run PCA using SageMaker
    pca = sagemaker.PCA(
        role=role,
        train_instance_count=1,
        train_instance_type='ml.c4.xlarge',
        output_path=s3_outdir,
        num_components=df_scaled.shape[1] - 1,  # number of principal components to be computed
        sagemaker_session=session
    )

    # Convert df to np.array.
    df_np = df_scaled.values.astype('float32')
    # Convert it to RecordSet.
    df_rs = pca.record_set(df_np)

    logger.info('PCA started.')
    pca.fit(df_rs)
    logger.info('PCA done.')


training_job_name = 'pca-2020-05-21-21-03-59-223'
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')
s3.download_file(
    Bucket=bucket,
    Key=model_key, 
    Filename=os.path.join(data_dir, 'model.tar.gz')
)

subprocess.check_call('tar -zxvf {}'.format(
    os.path.join(data_dir, 'model.tar.gz')
), shell=True)  # Untar in current directory

subprocess.check_call('mv model_algo-1 {}'.format(data_dir), shell=True)
model_artifacts = os.path.join(data_dir, 'model_algo-1')
pca_model_params = mx.ndarray.load(model_artifacts)
logger.info("Traind PCA model artifacts:\n{}".format(pca_model_params))


# Get selected parameters.
s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())