import os
import sagemaker
import boto3



def check_data_in_S3(data_dir, session, bucket, prefix, S3):
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
    s3_object_list = [content['Key'] for content in s3_object_dict['Contents']]

    local_data_list = os.listdir(data_dir)
    local_data_list = [os.path.join(prefix, f) for f in local_data_list]

    if set(local_data_list).intersection(s3_object_list) == set(local_data_list):
        input_data = session.upload_data(
            path=data_dir,
            bucket=bucket,
            key_prefix=prefix
        )
        print("Data uploaded to S3.")
        return(input_data)
    else:
        print("Data already present in S3.")
        pass


def evaluate(predictor, test_features, test_labels, verbose=True):
    prediction_batches = []
    for batch in np.array_split(test_features, 100):
        prediction_batches.append(predictor.predict(batch))

    pred_list_np = []
    for batch in prediction_batches:
        pred_list = []
        for x in batch:
            pred_list.append(x.label['predicted_label'].float32_tensor.values[0])
        pred_list_np.append(np.array(pred_list))

    test_preds = np.concatenate(pred_list_np)

    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()

    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)


    logger.info(
        "Linear Learner Metrics \n\
        Recall: {} \n\
        Precision: {} \n\
        Accuracy: {}\n".format(recall, precision, accuracy)
    )

    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
