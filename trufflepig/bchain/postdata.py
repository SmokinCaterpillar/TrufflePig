import logging
import time

from steem.post import Post, PostDoesNotExist, VotingInvalidOnArchivedPost
from steembase.exceptions import RPCError
from steem.converter import Converter

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd
import trufflepig.filters.textfilters as tftf
from trufflepig.utils import error_retry


logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'daily-truffle-picks-{date}'


def post_topN_list(sorted_post_frame, steem, account,
                   current_datetime, overview_permalink, N=10):
    """ Post the toplist to the blockchain

    Parameters
    ----------
    sorted_post_frame: DataFrame
    steem:  Steem object
    account: str
    current_datetime: datetime
    N: int
        Size of top list

    Returns
    -------
    permalink to new post

    """
    df = sorted_post_frame.iloc[:N, :]

    logger.info('Creating top {} post'.format(N))
    df.first_image_url = df.body.apply(lambda x: tftf.get_image_urls(x))

    steem_per_mvests = Converter(steem).steem_per_mvests()
    truffle_link = 'https://steemit.com/steemit/@{}/{}'.format(account,
                                                               overview_permalink)

    title, body = tfbp.topN_post(topN_authors=df.author,
                                 topN_permalinks=df.permalink,
                                 topN_titles=df.title,
                                 topN_filtered_bodies=df.filtered_body,
                                 topN_image_urls=df.first_image_url,
                                 topN_rewards=df.predicted_reward,
                                 topN_votes=df.predicted_votes,
                                 title_date=current_datetime,
                                 truffle_link=truffle_link,
                                 steem_per_mvests=steem_per_mvests)

    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%m-%d'))
    logger.info('Posting top post with permalink: {}'.format(permalink))
    logger.info(title)
    logger.info(body)
    error_retry(steem.commit.post)(author=account,
                                   title=title,
                                   body=body,
                                   permlink=permalink,
                                   self_vote=True,
                                   tags=tfbp.TAGS)

    return permalink


def comment_on_own_top_list(sorted_post_frame, steem, account,
                            topN_permalink, Kstart=10, Kend=25):
    """ Adds the more ranks as a comment

    Parameters
    ----------
    sorted_post_frame: DataFrame
    steem:  Steem object
    account: str
    topN_permalink: str
    Kstart: int
    Kend: int

    """
    df = sorted_post_frame.iloc[Kstart: Kend, :]

    comment = tfbp.topN_comment(topN_authors=df.author,
                                topN_permalinks=df.permalink,
                                topN_titles=df.title,
                                topN_rewards=df.predicted_reward,
                                topN_votes=df.predicted_votes,
                                nstart=Kstart + 1)


    logger.info('Commenting on top {} post with \n '
                '{}'.format(topN_permalink, comment))
    time.sleep(25)
    try:
        post = Post('@{}/{}'.format(account, topN_permalink), steem)
        # to pass around the no broadcast setting otherwise it is lost
        # see https://github.com/steemit/steem-python/issues/155
        post.commit.no_broadcast = steem.commit.no_broadcast
        error_retry(post.reply)(body=comment, author=account)
    except PostDoesNotExist:
        logger.exception('No broadcast, heh?')


def vote_and_comment_on_topK(sorted_post_frame, steem, account,
                             topN_permalink, overview_permalink, K=25):
    """

    Parameters
    ----------
    sorted_post_frame: DataFrame
    steem:  Steem object
    account: str
    topN_permalink: str
    K: int
        number of truffles to comment and upvote

    """
    logger.info('Voting and commenting on {} top truffles'.format(K))
    weight = min(800.0 / K, 100)
    topN_link = 'https://steemit.com/@{author}/{permalink}'.format(author=account,
                                                    permalink=topN_permalink)
    truffle_link = 'https://steemit.com/steemit/@{}/{}'.format(account,
                                                               overview_permalink)

    for kdx, (_, row) in enumerate(sorted_post_frame.iterrows()):
        if kdx >= K:
            break
        try:
            logger.info('Voting and commenting on https://steemit.com/@{author}/{permalink}'
                        ''.format(author=row.author, permalink=row.permalink))
            reply = tfbp.truffle_comment(reward=row.predicted_reward,
                                         votes=row.predicted_votes,
                                         rank=kdx + 1,
                                         topN_link=topN_link,
                                         truffle_link=truffle_link)
            post = Post('@{}/{}'.format(row.author, row.permalink), steem)
            # to pass around the no broadcast setting otherwise it is lost
            # see https://github.com/steemit/steem-python/issues/155
            post.commit.no_broadcast = steem.commit.no_broadcast

            # Wait a bit Steemit nodes hate comments in quick succession
            time.sleep(25)

            post.upvote(weight=weight, voter=account)
            post.reply(body=reply, author=account)
        except PostDoesNotExist:
            logger.exception('Post not found of row {}'.format(row))
        except VotingInvalidOnArchivedPost:
            logger.exception('Post archived of row {}'.format(row))
        except RPCError:
            logger.exception('Could not post row {}'.format(row))
        except Exception:
            logger.exception('W00t? row: {}'.format(row))


def create_wallet(steem, password, posting_key,
                  active_key=None):
    """ Creates a new wallet

    Does nothing if wallet database entry already exists

    Parameters
    ----------
    steem:  Steem object
    password: str
    posting_key: str
    active_key: str

    """
    if posting_key is None or password is None:
        raise RuntimeError('Key or password are None!')

    logger.info('Unlocking or creating wallet')
    wallet = steem.wallet
    wallet.unlock(pwd=password)

    logger.info('Adding POSTING Key')
    try:
        wallet.addPrivateKey(posting_key)
    except ValueError:
        logger.info('Key already present')

    if active_key:
        logger.info('Adding ACTIVE Key')
        try:
            wallet.addPrivateKey(active_key)
        except ValueError:
            logger.info('Key already present')

    logger.info('Wallet is ready')
