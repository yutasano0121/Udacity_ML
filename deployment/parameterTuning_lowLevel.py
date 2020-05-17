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
import time
from time import localtime, strftime

# sklearn and Boston dataset
import sklearn.model_selection
from sklearn.datasets import load_boston

# sagemaker libraries
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# hyperparameter tuning
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

# logger
import logging


# Create a logger.
working_dir = '/home/ec2-user/SageMaker/sagemaker-deployment/'
log_dir = working_dir + 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'boston_tutorial.log'

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
# currently assigned IAM role (i.e. access rights. See https://aws.amazon.com/iam/)
role = get_execution_role()

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
x_test.to_csv(
    os.path.join(data_dir, 'test.csv'),
    header=False,
    index=False
)

pd.concat([y_val, x_val], axis=1).to_csv(
    os.path.join(data_dir, 'validation.csv'),
    header=False,
    index=False
)

pd.concat([y_train, x_train], axis=1).to_csv(
    os.path.join(data_dir, 'train.csv'),
    header=False,
    index=False
)
logger.info('Data stored locally.')


# Upload the data to S3.
prefix = 'boston_xgboost_low'  # S3 folder name

test_location = session.upload_data(
    os.path.join(data_dir, 'test.csv'),
    key_prefix=prefix
)

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

# Define contaniner parameters
training_params = {
    'RoleArn': role,

    'AlgorithmSpecification': {
        'TrainingImage': container,
        'TrainingInputMode': 'File'
    },

    'ResourceConfig': {
        'InstanceCount': 1,
        'InstanceType': 'ml.m4.xlarge',
        'VolumeSizeInGB': 5
    },

    'StoppingCondition': {
        'MaxRuntimeInSeconds': 86400
    },

    'StaticHyperParameters': {
        'gamma': '4',
        'subsample': '0.8',
        'objective': 'reg:linear',
        'early_stopping_rounds': '10',
        'num_round': '200'
    },

    'InputDataConfig': [
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': train_location,
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'csv',
            'CompressionType': 'None'
        },
        {
            'ChannelName': 'validation',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': val_location,
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'csv',
            'CompressionType': 'None'
        }
    ],

    'OutputDataConfig': {
        'S3OutputPath': 's3://{}/{}/output'.format(
            session.default_bucket(),
            prefix
        )
    },
}


# Specify parameters to be tuned.
tuning_job_config = {
    'ParameterRanges': {
        'CategoricalParameterRanges': [],
        'ContinuousParameterRanges': [
            {
                'Name': 'eta',
                'MaxValue': '0.5',
                'MinValue': '0.05'
            }
        ],
        'IntegerParameterRanges': [
            {
                'Name': 'max_depth',
                'MaxValue': '12',
                'MinValue': '3'
            },
            {
                'Name': 'min_child_weight',
                'MaxValue': '8',
                'MinValue': '2'
            }
        ]
    },

    'ResourceLimits': {
        'MaxNumberOfTrainingJobs': 15,
        'MaxParallelTrainingJobs': 3
    },

    'Strategy': 'Bayesian',  # How to optimize the parameters.

    'HyperParameterTuningJobObjective': {
        'MetricName': 'validation:rmse',
        'Type': 'Minimize'
    }
}


# Specify the name of the training job
tuning_job_name = 'tuning' + strftime("%Y-%m-%d-%H-%M-%S", localtime())

# Execute the training job.
session.sagemaker_client.create_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name,
    HyperParameterTuningJobConfig=tuning_job_config,
    TrainingJobDefinition=training_params
)

logger.info(
    "Training job created. \n \
    Name: {} \n \
    JobConfig: {} \n \
    JobDefinition: {}".format(
        tuning_job_name,
        tuning_job_config,
        training_params
    )
)

# Wait until the training is done.
session.wait_for_tuning_job(tuning_job_name)
logger.info("Training done.")

# Fetch the best training job.
training_job_info = session.sagemaker_client.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name
)

best_training_job_name = training_job_info['BestTrainingJob']['TrainingJobName']
training_job_info = session.sagemaker_client.describe_training_job(
    TrainingJobName=best_training_job_name
)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']
logger.info(
    "Best model: {} \n \
    Artifacts: {}".format(
        best_training_job_name,
        model_artifacts
    )
)

# Assign a name to the best model.
model_name = best_training_job_name + 'model'

# Assign a container for prediction.
# Use the same container used for training.
primary_container = {
    'Image': container,
    'ModelDataUrl': model_artifacts
}

# Construct a model.
model_info = session.sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=primary_container
)

"""
# BELOW IS FOR A HIGH-LEVEL API
# Construct a XGBoost extimator.
xgb = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
    sagemaker_session=session
)

# Set initial hyperparameters.
xgb.set_hyperparameters(
    max_depth=5,  # default = 5. Lower values regularize overfitting.
    eta=0.2,  # learning rate. default = 0.3.
    gamma=4,  # the minimum loss required to make a split. default = 0.
    # minimum sum of weights for all observations in a child.
    min_child_weight=6,
    # default = 1. Higher values regularize overfitting.
    # fraction of samples randomely sampled for each tree. default = 1.
    subsample=0.8,
    objective='reg:linear',
    early_stopping_rounds=10,
    num_round=200
)

# Tune hyperparameters.
xgb_tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name='validation:rmse',
    objective_type='Minimize',
    max_jobs=21,
    max_parallel_jobs=3,
    hyperparameter_ranges={
        'max_depth': IntegerParameter(3, 12),
        'eta': ContinuousParameter(0.05, 0.5),
        'min_child_weight': IntegerParameter(2, 8),
        'subsample': ContinuousParameter(0.5, 0.9),
        'gamma': ContinuousParameter(0, 10)
    }
)


# Specify the data location in S3.
s3_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_val = sagemaker.s3_input(s3_data=val_location, content_type='csv')


# Fit the model.
logger.info('Model fitting')
xgb_tuner.fit({
    'train': s3_train,
    'validation': s3_val
})

xgb_tuner.wait()

# Retrieve the best model.
xgb_tuner.best_training_job()

# Attach it to use it as an estimator.
xgb_attached = sagemaker.estimator.Estimator.attach(
    xgb_tuner.best_training_job())
"""


"""
Deploy the trained model to make an estimate.
Do not forget to shut down a deployed model after run
by "DEPLOYED_MODEL.delete_endpoint."
"""


logger.info('Batch-transform instead of making an endpoint.')

trainsform_job_name = 'boston-transform-' + strftime("%Y-%m-%d-%H-%M-%S", localtime())
transform_request = {
    'TransformJobName': trainsform_job_name,
    'ModelName': model_name,
    'MaxConcurrentTransforms': 1,
    'MaxPayloadInMB': 6,
    'BatchStrategy': 'MultiRecord',
    'TransformOutput': {
        'S3OutputPath': "s3://{}/{}/batch-bransform/".format(
            session.default_bucket(),
            prefix
        )
    },
    'TransformInput': {
        'ContentType': 'text/csv',
        'SplitType': 'Line',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': test_location
            }
        }
    },
    'TransformResources': {
        'InstanceType': 'ml.m4.xlarge',
        'InstanceCount': 1
    }
}

logger.info("Initiate the transform job.")
transform_response = session.sagemaker_client.create_transform_job(
    **transform_request
)
transform_desc = session.wait_for_transform_job(trainsform_job_name)
logger.info("Transform done.")



"""
# For a high-level API
xgb_transformer = xgb_attached.transformer(
    instance_count=1,
    instance_type='ml.m4.xlarge'
)

xgb_transformer.transform(
    test_location,
    content_type='text/csv',
    split_type='Line'
)

xgb_transformer.wait()
"""


# Copy results from S3 to the local
subprocess.check_call("aws s3 cp --recursive {} {}".format(
    xgb_transformer.output_path,
    data_dir
    ), shell=True
)

y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header = None)


# Plot predicted/true values.
plt.scatter(y_test, y_pred)
plot.xlabel('true')
plot.ylabel('predicted')



# Remove locally stored files.
subprocess.check_call('rm {}/*'.format(data_dir), shell=True)
logger.info('Local file deleted. All done.')
