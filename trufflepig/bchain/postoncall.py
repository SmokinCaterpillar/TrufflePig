import logging
import time

from steem.post import Post

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd
from trufflepig.utils import rpcerror_retry


logger = logging.getLogger(__name__)


I_WAS_HERE = 'Huh? Seems like I already voted on this post, thanks for calling anyway!'

YOU_DID_NOT_MAKE_IT = """I am sorry, I cannot evaluate your post. This can have several reasons, for example, it may not be long enough, it's not in English, or has been filtered, etc."""


def post_on_call(post_frame, account, steem, topN_link,
                 overview_permalink,
                 exclusion_set=tfgd.EXCLUSION_VOTERS_SET,
                 sleep_time=20.1):
    """ Replies to users calling @trufflepig

    Parameters
    ----------
    post_frame: DataFrame
    account: str
    steem: Steem object
    topN_link: str
    exclusion_set: set of str
    sleep_time: float
        Bot can only post every 20 seconds,
        should only be lowered for debugging

    """
    weight = min(75 / len(post_frame), 10)
    truffle_link = 'https://steemit.com/steemit/@{}/{}'.format(account,
                                                               overview_permalink)

    for kdx, (_, row) in enumerate(post_frame.iterrows()):
        try:
            comment = Post('@{}/{}'.format(row.comment_author,
                                           row.comment_permalink), steem)
            comment.commit.no_broadcast = steem.commit.no_broadcast
            # Wait a bit Steemit nodes hate comments in quick succession
            time.sleep(sleep_time)
            if not tfgd.exclude_if_voted_by(row.active_votes, {account}):
                if row.passed and not tfgd.exclude_if_voted_by(row.active_votes, exclusion_set):

                        logger.info('Voting and commenting on https://steemit.com/@{author}/{permalink}'
                                        ''.format(author=row.author, permalink=row.permalink))
                        reply = tfbp.on_call_comment(reward=row.predicted_reward,
                                                     author=row.comment_author,
                                                     votes=row.predicted_votes,
                                                     topN_link=topN_link,
                                                     truffle_link=truffle_link)

                        post = Post('@{}/{}'.format(row.author, row.permalink), steem)
                        # to pass around the no broadcast setting otherwise it is lost
                        # see https://github.com/steemit/steem-python/issues/155
                        post.commit.no_broadcast = steem.commit.no_broadcast

                        # We cannot use this post.upvote(weight=weight, voter=account)
                        # because we need to vote on archived posts as a flag!
                        rpcerror_retry(post.commit.vote)(post.identifier, weight, account=account)
                else:
                    reply = YOU_DID_NOT_MAKE_IT
            else:
                reply = I_WAS_HERE

            replies = comment.steemd.get_content_replies(row.comment_author,
                                                         row.comment_permalink)
            reply_authors = set(x['author'] for x in replies)
            if account not in reply_authors:
                logger.info('Replying to https://steemit.com/@{author}/{permalink} '
                            'with {answer}...'.format(author=row.comment_author,
                                                   permalink=row.comment_permalink,
                                                   answer=reply[:256]))
                rpcerror_retry(comment.reply)(body=reply, author=account)
            else:
                logger.info('Already answered {} by {}, will '
                            'skip!'.format(row.comment_author,
                                           row.comment_permalink))

        except Exception as e:
            logger.exception('Something went wrong with row {}'.format(row))