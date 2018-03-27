import logging
import time

from steem.post import Post

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd
import trufflepig.filters.textfilters as tftf
from trufflepig.utils import error_retry
import trufflepig.preprocessing as tppp
from trufflepig.bchain.poster import Poster


logger = logging.getLogger(__name__)


I_WAS_HERE = 'Huh? Seems like I already voted on this post, thanks for calling anyway!'

YOU_DID_NOT_MAKE_IT = """I am sorry, I cannot evaluate your post. This can have several reasons, for example, it may not be long enough, it's not in English, or has been filtered, etc."""


def post_on_call(post_frame, topN_link,
                 poster,
                 overview_permalink,
                 filter_voters=tppp.FILTER_VOTERS):
    """ Replies to users calling @trufflepig

    Parameters
    ----------
    post_frame: DataFrame
    poster: Poster
    topN_link: str
    filter_voters: set of str

    """
    weight = min(75 / len(post_frame), 10)
    truffle_link = 'https://steemit.com/steemit/@{}/{}'.format(poster.account,
                                                               overview_permalink)

    for kdx, (_, row) in enumerate(post_frame.iterrows()):
        try:
            comment = Post('@{}/{}'.format(row.comment_author,
                                           row.comment_permalink), poster.steem)

            if not tftf.voted_by(row.active_votes, {poster.account}):
                if row.passed and not tftf.voted_by(row.active_votes, filter_voters):

                        logger.info('Voting and commenting on https://steemit.com/@{author}/{permalink}'
                                        ''.format(author=row.author, permalink=row.permalink))
                        reply = tfbp.on_call_comment(reward=row.predicted_reward,
                                                     author=row.comment_author,
                                                     votes=row.predicted_votes,
                                                     topN_link=topN_link,
                                                     truffle_link=truffle_link)

                        poster.vote(row.author, row.permalink, weight)
                else:
                    reply = YOU_DID_NOT_MAKE_IT
            else:
                reply = I_WAS_HERE

            replies = comment.steemd.get_content_replies(row.comment_author,
                                                         row.comment_permalink)
            reply_authors = set(x['author'] for x in replies)
            if poster.account not in reply_authors:
                poster.reply(body=reply,
                             parent_author=row.comment_author,
                             parent_permalink=row.comment_permalink)
            else:
                logger.info('Already answered {} by {}, will '
                            'skip!'.format(row.comment_author,
                                           row.comment_permalink))

        except Exception as e:
            logger.exception('Something went wrong with row {}.'
                             'Reconnecting...'.format(row))
            poster.steem.reconnect()