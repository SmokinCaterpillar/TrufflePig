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


def call_a_pig(steem_kwargs, account, pipeline, topN_link, current_datetime,
               offset_hours=2, hours=24, max_comments=MAX_COMMENTS,
               sleep_time=20.1):

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

        truffle_frame = tpmo.find_truffles(post_frame, pipeline, k=0,
                                           account='', add_rank_score=False)
        truffle_frame['passed'] = True

        filtered_posts = initial_frame[~initial_frame.index.isin(truffle_frame.index)]
        filtered_posts['passed'] = False

        combined = pd.concat([truffle_frame, filtered_posts], axis=0)

        tpoc.post_on_call(combined, account=account,
                              steem=steem,
                              topN_link=topN_link,
                              sleep_time=sleep_time)
    else:
        logger.info('No mentions of {} found, good bye!'.format(account))
