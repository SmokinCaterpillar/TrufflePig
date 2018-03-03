import argparse
import concurrent
import gc
import logging
import os
import time

import pandas as pd

import trufflepig.bchain.getdata as tpgd
import trufflepig.bchain.postdata as tppd
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.utils as tfut
import trufflepig.pigonduty as tfod
from trufflepig import config
from trufflepig.utils import configure_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='TrufflePig Bot')
    parser.add_argument('--broadcast', action="store_false",
                        default=True)
    parser.add_argument('--now', action='store', default=None)
    args = parser.parse_args()
    return args.broadcast, args.now


def large_mp_preprocess(log_directory, current_datetime, steem_kwargs, data_directory,
                        days, offset_days):
    """Helper function to spawn in child process"""
    configure_logging(log_directory, current_datetime)
    post_frame = tpgd.load_or_scrape_training_data(steem_kwargs, data_directory,
                                                       current_datetime=current_datetime,
                                                       days=days,
                                                       offset_days=offset_days,
                                                       ncores=20)
    return tppp.preprocess(post_frame, ncores=4)


def load_and_preprocess_2_frames(log_directory, current_datetime, steem_kwargs,
                                 data_directory, offset_days=8,
                                 days=5, days2=5):
    """ Function to load and preprocess the time span split into 2
    for better memory footprint

    Parameters
    ----------
    log_directory: str
    current_datetime: datetime
    steem_kwargs: dict
    data_directory: str
    offset_days: int
    days: int
    days2: int
    ncores: int

    Returns
    -------
    DataFrame

    """
    # hack for better memory footprint,
    # see https://stackoverflow.com/questions/15455048/releasing-memory-in-python
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        post_frame = executor.submit(large_mp_preprocess,
                                     log_directory=log_directory,
                                     current_datetime=current_datetime,
                                     steem_kwargs=steem_kwargs,
                                     data_directory=data_directory,
                                     days=days,
                                     offset_days=offset_days).result()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        post_frame2 = executor.submit(large_mp_preprocess,
                                     log_directory=log_directory,
                                     current_datetime=current_datetime,
                                     steem_kwargs=steem_kwargs,
                                     data_directory=data_directory,
                                     days=days2,
                                     offset_days=8 + days).result()

    post_frame = pd.concat([post_frame, post_frame2], axis=0)
    # We need to reset the index because due to concatenation
    # the default indices are duplicates!
    post_frame.reset_index(inplace=True, drop=True)
    logger.info('Combining 2 frames into 1')
    post_frame = tppp.filter_duplicates(post_frame)
    return post_frame


def main():
    """Main loop started from command line"""

    no_broadcast, current_datetime = parse_args()

    if current_datetime is None:
        current_datetime = pd.datetime.utcnow()
    else:
        current_datetime = pd.to_datetime(current_datetime)

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

        post_frame = load_and_preprocess_2_frames(
            log_directory=log_directory,
            current_datetime=current_datetime,
            steem_kwargs=steem_kwargs,
            data_directory=data_directory
        )
        logger.info('Garbage collecting')
        gc.collect()
    else:
        post_frame = None

    regressor_kwargs = dict(n_estimators=256, max_leaf_nodes=5000,
                              max_features=0.2, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=128, no_below=7, no_above=0.1,
                        ngrams=(1,2), keep_n=300000)

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
                                             ncores=20,
                                             offset_hours=2)
    prediction_frame = tppp.preprocess(prediction_frame, ncores=3)

    sorted_frame = tpmo.find_truffles(prediction_frame, pipeline,
                                      account=config.ACCOUNT)
    account = config.ACCOUNT
    permalink = tppd.post_topN_list(sorted_frame, steem_kwargs,
                                    account=account,
                                    current_datetime=current_datetime)

    tppd.comment_on_own_top_list(sorted_frame, steem_kwargs,
                                 account=account,
                                 topN_permalink=permalink)

    tppd.vote_and_comment_on_topK(sorted_frame,
                                  steem_kwargs,
                                  topN_permalink=permalink,
                                  account=account)

    logger.info('Done with normal duty, answering manual calls!')
    tfod.call_a_pig(steem_kwargs=steem_kwargs,
                    account=account,
                    pipeline=pipeline,
                    topN_link=permalink,
                    current_datetime=current_datetime,
                    offset_hours=2,
                    hours=24)

    logger.info('Cleaning up after myself')
    tfut.clean_up_directory(model_directoy, keep_last=3)
    tfut.clean_up_directory(data_directory, keep_last=25)
    tfut.clean_up_directory(log_directory, keep_last=14)

    logger.info('DONE at {}'.format(current_datetime))


if __name__ == '__main__':
    main()
