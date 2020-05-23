import numpy as np
import pandas as pd
import sagemaker


def print_fraudRatio(data):
    fraud_num = (data.Class == 1).sum()
    print(
        "Fraudulent cases: {}\
        Valid cases: {}\
        Fraudulent ratio: {}".format(
            fraud_num,
            data.shape[0] - fraud_num,
            fraud_num / data.shape[0]
        )
    )
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

    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

print('Metrics for simple, LinearLearner.\n')
