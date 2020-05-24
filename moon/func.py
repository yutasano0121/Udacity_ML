import os
import sagemaker
import boto3
import numpy as np
import pandas as pd



def check_data_in_S3(data_dir, session, bucket, prefix, S3, fname='train.csv'):
    """
    data_dir: local data directory
    session: sagemaker.Session()
    bucket: sagemaker.Session().default_bucket()
    prefix: S3 path prefix
    S3: boto3.client('s3')
    """
    s3_object_dict = S3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix
    )
    try:  # If the s3 folder does not exist 'Contents' returns KeyError, so use 'try-except.'
        s3_object_list = [content['Key'] for content in s3_object_dict['Contents']]

        training_data = os.path.join(prefix, fname) 

        if not training_data in s3_object_list:
            input_data = session.upload_data(
                path=os.path.join(data_dir, fname),
                bucket=bucket,
                key_prefix=prefix
            )
            print("Data uploaded to S3.")
            return(input_data)
        else:
            print("Data already present in S3.")
            input_data = 's3://{}/{}'.format(bucket, training_data)
            return(input_data)
    except KeyError:
        input_data = session.upload_data(
            path=os.path.join(data_dir, fname),
            bucket=bucket,
            key_prefix=prefix
        )
        print("Data uploaded to S3.")
        return(input_data)


def evaluate(predictor, test_features, test_labels, verbose=True):
    # rounding and squeezing array
    test_preds = np.squeeze(np.round(predictor.predict(test_features)))

    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()

    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)


    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
