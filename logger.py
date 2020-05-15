import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:(name)s:%(message)s')

try:
    file_handler = logging.FileHandler(log_filename)
    print(
        'log file {} created'.format(log_filename)
    )
except(NameError):
    file_handler = logging.FileHandler('test.log')
    print(
        'No log filename specified. \
        Test.log created in the current working directory.'
    )
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
