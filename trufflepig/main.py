import logging
import os
import gc

import pandas as pd

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
import trufflepig.bchain.postdata as tppd
import trufflepig.utils as tfut
from trufflepig import config
import argparse
import time


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='TrufflePig Bot')
    parser.add_argument('--broadcast', action="store_false",
                        default=True)
    args = parser.parse_args()
    return args.broadcast


def configure_logging(directory, current_datetime):
    if not os.path.isdir(directory):
        os.makedirs(directory)

    filename = 'trufflepig_{time}.txt'.format(time=current_datetime.strftime('%Y-%m-%d'))
    filename = os.path.join(directory, filename)

    format=('%(asctime)s %(processName)s:%(name)s:'
                  '%(funcName)s:%(lineno)s:%(levelname)s: %(message)s')
    handlers = [logging.StreamHandler(), logging.FileHandler(filename)]
    logging.basicConfig(level=logging.INFO, format=format,
                        handlers=handlers)


def main():

    no_broadcast = parse_args()

    current_datetime = pd.datetime.utcnow()

    data_directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')
    model_directoy = os.path.join(config.PROJECT_DIRECTORY, 'trained_models')
    log_directory =  os.path.join(config.PROJECT_DIRECTORY, 'logs')

    configure_logging(log_directory, current_datetime)

    logger.info('STARTING main script at {}'.format(current_datetime))
    if no_broadcast:
        logger.info('Run without broadcasting.')
    else:
        logger.info('ATTENTION I WILL BROADCAST TO STEEMIT!!!')
    time.sleep(2)

    steem_kwargs = dict(nodes=config.NODES, no_broadcast=no_broadcast)

    tppd.create_wallet(steem_kwargs, config.PASSWORD, config.POSTING_KEY)

    if not tpmo.model_exists(current_datetime, model_directoy):
        post_frame = tpgd.load_or_scrape_training_data(steem_kwargs, data_directory,
                                                       current_datetime=current_datetime,
                                                       days=7,
                                                       offset_days=8,
                                                       ncores=16)

        post_frame = tppp.preprocess(post_frame, ncores=3)
        logger.info('Garbage collecting')
        gc.collect()
    else:
        post_frame = None

    regressor_kwargs = dict(n_estimators=256, max_leaf_nodes=5000,
                              max_features=0.2, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=128, no_below=5, no_above=0.1)

    pipeline = tpmo.load_or_train_pipeline(post_frame, model_directoy,
                                           current_datetime,
                                           regressor_kwargs=regressor_kwargs,
                                           topic_kwargs=topic_kwargs)

    tpmo.log_pipeline_info(pipeline=pipeline)

    logger.info('Garbage collecting')
    del post_frame
    gc.collect()

    prediction_frame = tpgd.scrape_hour_data(steem_or_args=steem_kwargs,
                                             current_datetime=current_datetime,
                                             ncores=16)
    prediction_frame = tppp.preprocess(prediction_frame, ncores=3)

    sorted_frame = tpmo.find_truffles(prediction_frame, pipeline)
    account = config.ACCOUNT
    permalink = tppd.post_topN_list(sorted_frame, steem_kwargs,
                                    account=account,
                                    current_datetime=current_datetime)
    tppd.vote_and_comment_on_topK(sorted_frame,
                                  steem_kwargs,
                                  topN_permalink=permalink,
                                  account=account)

    logger.info('Cleaning up after myself')
    tfut.clean_up_directory(model_directoy, keep_last=3)
    tfut.clean_up_directory(data_directory, keep_last=25)
    tfut.clean_up_directory(log_directory, keep_last=14)
    logger.info('DONE at {}'.format(current_datetime))


if __name__ == '__main__':
    main()
