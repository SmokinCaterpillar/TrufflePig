"""Module to allow manual calling of @trufflepig"""

import logging

import pandas as pd

import trufflepig.bchain.checkops as tpco
import trufflepig.bchain.getdata as tpbg
import trufflepig.preprocessing as tppp
import trufflepig.model as tpmo
import trufflepig.bchain.postoncall as tpoc


logger = logging.getLogger(__name__)


MAX_COMMENTS = 3000


def call_a_pig(steem_kwargs, account, pipeline, topN_permalink, current_datetime,
               offset_hours=2, hours=24, max_comments=MAX_COMMENTS,
               sleep_time=20.1):
    """ Scans for user mentioning the bot and answers

    Parameters
    ----------
    steem_kwargs: dict
    account: str
    pipeline: sklearn pipeline
    topN_link: str
    current_datetime: datetime
    offset_hours: int
    hours: int
    max_comments: int
    sleep_time: float

    """

    steem = tpbg.check_and_convert_steem(steem_kwargs)

    current_datetime = pd.to_datetime(current_datetime)

    end_datetime = current_datetime - pd.Timedelta(hours=offset_hours)
    start_datetime = end_datetime - pd.Timedelta(hours=hours)

    logger.info('Scanning for mentions of {} between {} and '
                '{}'.format(account, start_datetime, end_datetime))

    comment_authors_and_permalinks = tpco.check_all_ops_between_parallel(
        account=account,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        steem_args=steem_kwargs,
        ncores=20
    )

    if comment_authors_and_permalinks:
        execute_call(comment_authors_and_permalinks=comment_authors_and_permalinks,
                     account=account,
                     pipeline=pipeline,
                     steem=steem,
                     topN_permalink=topN_permalink,
                     sleep_time=sleep_time,
                     max_comments=max_comments)
    else:
        logger.info('No mentions of {} found, good bye!'.format(account))


def execute_call(comment_authors_and_permalinks, account, pipeline,
                 steem, topN_permalink, sleep_time, max_comments):
    """Executes the pig on duty call"""
    ncomments = len(comment_authors_and_permalinks)

    logger.info('Found {} comments mentioning {}'.format(ncomments,
                                                         account))
    if ncomments > max_comments:
        logger.info('To many comments, reducing to {}'.format(max_comments))
        comment_authors_and_permalinks = comment_authors_and_permalinks[:max_comments]

    posts = tpco.get_parent_posts(comment_authors_and_permalinks, steem)

    initial_frame = pd.DataFrame(posts)
    post_frame = initial_frame.copy()

    post_frame = tppp.preprocess(post_frame, ncores=4)

    if len(post_frame):
        truffle_frame = tpmo.find_truffles(post_frame, pipeline, k=0,
                                           account='', add_rank_score=False)
        truffle_frame['passed'] = True
    else:
        truffle_frame = pd.DataFrame()

    filtered_posts = initial_frame[~initial_frame.index.isin(truffle_frame.index)]
    filtered_posts['passed'] = False

    combined = pd.concat([truffle_frame, filtered_posts], axis=0)

    topN_link = 'https://steemit.com/@{author}/{permalink}'.format(author=account,
                                                    permalink=topN_permalink)

    tpoc.post_on_call(combined, account=account,
                          steem=steem,
                          topN_link=topN_link,
                          sleep_time=sleep_time)
