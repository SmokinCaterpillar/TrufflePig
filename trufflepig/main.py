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
import trufflepig.bchain.paydelegates as tpde
import trufflepig.bchain.getaccountdata as tpad
from trufflepig import config
from trufflepig.utils import configure_logging
import trufflepig.bchain.postweeklyupdate as tppw
from trufflepig.bchain.mpsteem import MPSteem
from trufflepig.bchain.poster import Poster
import trufflepig.trending0bidbots as tt0b


logger = logging.getLogger(__name__)


MAX_DOCUMENTS = 123000


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='TrufflePig Bot')
    parser.add_argument('--broadcast', action="store_false",
                        default=True)
    parser.add_argument('--now', action='store', default=None)
    args = parser.parse_args()
    return args.broadcast, args.now


def large_mp_preprocess(log_directory, current_datetime, steem, data_directory,
                        days, offset_days):
    """Helper function to spawn in child process"""
    configure_logging(log_directory, current_datetime)
    post_frame = tpgd.load_or_scrape_training_data(steem, data_directory,
                                                       current_datetime=current_datetime,
                                                       days=days,
                                                       offset_days=offset_days,
                                                       ncores=32)
    return tppp.preprocess(post_frame, ncores=8)


def load_and_preprocess_2_frames(log_directory, current_datetime, steem,
                                 data_directory, offset_days=8,
                                 days=7, days2=7):
    """ Function to load and preprocess the time span split into 2
    for better memory footprint

    Parameters
    ----------
    log_directory: str
    current_datetime: datetime
    steem: MPSteem
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
                                     steem=steem,
                                     data_directory=data_directory,
                                     days=days,
                                     offset_days=offset_days).result()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        post_frame2 = executor.submit(large_mp_preprocess,
                                     log_directory=log_directory,
                                     current_datetime=current_datetime,
                                     steem=steem,
                                     data_directory=data_directory,
                                     days=days2,
                                     offset_days=offset_days + days).result()

    post_frame = pd.concat([post_frame, post_frame2], axis=0)
    # We need to reset the index because due to concatenation
    # the default indices are duplicates!
    post_frame.reset_index(inplace=True, drop=True)
    logger.info('Combining 2 frames into 1')
    post_frame = tppp.filter_duplicates(post_frame)

    logger.info('Searching for bid bots and bought votes')
    min_datetime = post_frame.created.min()
    max_datetime = post_frame.created.max() + pd.Timedelta(days=8)
    upvote_payments = tpad.get_upvote_payments_to_bots(steem=steem,
                                                  min_datetime=min_datetime,
                                                  max_datetime=max_datetime)
    logger.info('Adjusting votes and reward')
    post_frame = tppp.compute_bidbot_correction(post_frame=post_frame,
                                                upvote_payments=upvote_payments)
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

    steem = MPSteem(nodes=config.NODES, no_broadcast=no_broadcast)
    # hack to allow for payments, because of https://github.com/steemit/steem-python/issues/191
    noapisteem = MPSteem(nodes=config.NODES[1:], no_broadcast=no_broadcast)
    # To post stuff
    account = config.ACCOUNT
    poster = Poster(account=account, steem=steem)

    tppd.create_wallet(steem, config.PASSWORD,
                       posting_key=config.POSTING_KEY,
                       active_key=config.ACTIVE_KEY)

    logger.info('Paying out investors')
    tpde.pay_delegates(account=account,
                       steem=noapisteem, # use a steem instance without api.steem!
                       current_datetime=current_datetime)

    if not tpmo.model_exists(current_datetime, model_directoy):

        post_frame = load_and_preprocess_2_frames(
            log_directory=log_directory,
            current_datetime=current_datetime,
            steem=steem,
            data_directory=data_directory)
        logger.info('Garbage collecting')
        gc.collect()
    else:
        post_frame = None

    regressor_kwargs = dict(n_estimators=256, max_leaf_nodes=5000,
                              max_features=0.2, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=128, no_below=7, no_above=0.1,
                        ngrams=(1,2), keep_n=333000)

    if post_frame is not None and len(post_frame) > MAX_DOCUMENTS:
        logger.info('Frame has {} Documents, too many, '
                    'reducing to {}'.format(len(post_frame), MAX_DOCUMENTS))
        post_frame.sort_values('created', inplace=True, ascending=False)
        train_frame = post_frame.iloc[:MAX_DOCUMENTS, :]
    else:
        train_frame = post_frame

    pipeline = tpmo.load_or_train_pipeline(train_frame, model_directoy,
                                           current_datetime,
                                           regressor_kwargs=regressor_kwargs,
                                           topic_kwargs=topic_kwargs,
                                           targets=['adjusted_reward', 'adjusted_votes'])

    tpmo.log_pipeline_info(pipeline=pipeline)

    overview_permalink = tppw.return_overview_permalink_if_exists(account=account,
                                                                  current_datetime=current_datetime,
                                                                  steem=steem)

    if not overview_permalink:
        if post_frame is None:
            logger.info('Need to reaload data for weekly overview')
            post_frame = load_and_preprocess_2_frames(
            log_directory=log_directory,
            current_datetime=current_datetime,
            steem=steem,
            data_directory=data_directory)

        logger.info('I want to post my weekly overview')
        overview_permalink = tppw.post_weakly_update(pipeline=pipeline,
                                                     post_frame=post_frame,
                                                     poster=poster,
                                                     current_datetime=current_datetime)

    logger.info('Garbage collecting')
    del post_frame
    gc.collect()

    prediction_frame = tpgd.scrape_hour_data(steem=steem,
                                             current_datetime=current_datetime,
                                             ncores=32,
                                             offset_hours=2)
    prediction_frame = tppp.preprocess(prediction_frame, ncores=8)

    sorted_frame = tpmo.find_truffles(prediction_frame, pipeline,
                                      account=account)


    permalink = tppd.post_topN_list(sorted_frame, poster=poster,
                                    current_datetime=current_datetime,
                                    overview_permalink=overview_permalink)

    tppd.comment_on_own_top_list(sorted_frame, poster=poster,
                                 topN_permalink=permalink)

    tppd.vote_and_comment_on_topK(sorted_frame,
                                  poster=poster,
                                  topN_permalink=permalink,
                                  overview_permalink=overview_permalink)

    if current_datetime.day % 2 == 1:
        logger.info('Computing the top trending without bidbots')
        logger.info('Searching for bid bots and bought votes')
        min_datetime = sorted_frame.created.min()
        max_datetime = sorted_frame.created.max() + pd.Timedelta(days=1)
        upvote_payments = tpad.get_upvote_payments_to_bots(steem=steem,
                                                      min_datetime=min_datetime,
                                                      max_datetime=max_datetime)
        logger.info('Adjusting votes and reward')
        sorted_frame = tppp.compute_bidbot_correction(post_frame=sorted_frame,
                                                    upvote_payments=upvote_payments)
        tt0b.create_trending_post(sorted_frame,
                                  upvote_payments=upvote_payments,
                                  poster=poster,
                                  topN_permalink=permalink,
                                  overview_permalink=overview_permalink,
                                  current_datetime=current_datetime)
    else:
        logger.info('Computing community vote rep top list')
        sorted_frame = tppp.compute_reputation_vote_score(sorted_frame)
        tt0b.create_contributor_top_post(sorted_frame,
                                         poster=poster,
                                         topN_permalink=permalink,
                                         overview_permalink=overview_permalink,
                                         current_datetime=current_datetime)

    logger.info('Done with normal duty, answering manual calls!')
    tfod.call_a_pig(poster=poster,
                    pipeline=pipeline,
                    topN_permalink=permalink,
                    current_datetime=current_datetime,
                    offset_hours=2,
                    hours=24,
                    overview_permalink=overview_permalink)

    logger.info('Cleaning up after myself')
    tfut.clean_up_directory(model_directoy, keep_last=3)
    tfut.clean_up_directory(data_directory, keep_last=25)
    tfut.clean_up_directory(log_directory, keep_last=14)

    logger.info('Preloading -8 days for later training')
    tpgd.load_or_scrape_training_data(steem, data_directory,
                                                       current_datetime=current_datetime,
                                                       days=1,
                                                       offset_days=8,
                                                       ncores=32)

    logger.info('DONE at {}'.format(current_datetime))


if __name__ == '__main__':
    main()
