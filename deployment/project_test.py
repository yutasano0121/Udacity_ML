import glob
from tqdm import tqdm
import logging


logger = logging.getLogger('test')


def test_reviews(predictor, data_dir, stop=250):
    results=[]
    ground=[]

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
                results.append(
                    int(predictor.predict(review_input))  # Use an endpoint.
                )

            files_read += 1
            if files_read == stop:
                break

    # Return lists of ground truths and prediction results.
    return ground, results
