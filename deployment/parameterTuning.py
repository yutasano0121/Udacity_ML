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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


"""
Determine if a persistent endpoint is needed.
"""
need_endpoint = False


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
    header=False, index=False
)

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
Deploy the trained model to make an estimate.
Do not forget to shut down a deployed model after run
by "DEPLOYED_MODEL.delete_endpoint."
"""

if need_endpoint is True:
    # Create an endpoint.
    logger.info('Endpoint created.')
    xgb_predictor = xgb.deploy(
        initial_instance_count=1,
        instance_type='ml.m4.xlarge'
    )

    # Specify the format of input data.
    xgb_predictor.content_type = 'text/csv'
    xgb_predictor.serializer = csv_serializer

    # make a prediction
    logger.info('Making a prediction.')
    y_pred = xgb_predictor.predict(x_test.values).decode(
        'utf-8')  # Returns a string
    y_pred = np.fromstring(y_pred, sep=',')  # Make it to np.array.

    # Close the endpoint.
    xgb_predictor.delete_endpoint()
    logger.info('Endpoint deleted.')

else:
    logger.info('Batch-transform instead of making an endpoint.')

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

    # Copy results from S3 to the local
    !aws s3 cp --recursive $xgb_transformer.output_path $data_dir

    y_pred = pd.read_csv(os.path.join(data)dir, 'test.csv.out'), header = None)


# Plot predicted/true values.
plt.scatter(y_test, y_pred)
plot.xlabel('true')
plot.ylabel('predicted')



# Remove locally stored files.
os.system('rm {}/*'.format(data_dir))
logger.info('Local file deleted. All done.')
