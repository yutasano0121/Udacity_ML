import glob
import numpy as np
import os
from tqdm import tqdm
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import RealTimePredictor
import logging


logger = logging.getLogger('test')


def test_reviews(predictor, data_dir, stop=250):
    results = []
    ground = []

    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)  # a list of files to read

        files_read = 0  # number of files read

        logger.info('Starting ', sentiment, ' files')

        for f in tqdm(files):
            with open(f) as review:
                if sentiment == 'pos':
                    ground.append(1)
                else:
                    ground.append(0)
                review_input = review.read().encode('utf-8')
                result = predictor.predict(review_input)
                results.append(int(result))

            files_read += 1
            if files_read == stop:
                break

    # Return lists of ground truths and prediction results.
    return ground, results


class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            content_type='text/plain'
        )


def predict(data, deployed_model, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, deployed_model.predict(array))
