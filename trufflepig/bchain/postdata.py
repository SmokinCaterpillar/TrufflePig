import logging
import time

from steem.post import Post, PostDoesNotExist, VotingInvalidOnArchivedPost
from steembase.exceptions import RPCError

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd


logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'daily-truffle-picks-{date}'


def post_topN_list(sorted_post_frame, steem_or_args, account,
                   current_datetime, N=10):
    """ Post the toplist to the blockchain

    Parameters
    ----------
    sorted_post_frame: DataFrame
    steem_or_args: kwargs or Steem object
    account: str
    current_datetime: datetime
    N: int
        Size of top list

    Returns
    -------
    permalink to new post

    """
    steem = tfgd.check_and_convert_steem(steem_or_args)

    df = sorted_post_frame.iloc[:N, :]

    logger.info('Creating top {} post'.format(N))
    title, body = tfbp.topN_post(topN_authors=df.author,
                             topN_permalinks=df.permalink,
                             topN_titles=df.title,
                             topN_filtered_bodies=df.filtered_body,
                             topN_votes=df.predicted_votes,
                             topN_rewards=df.predicted_reward,
                             title_date=current_datetime)

    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%m-%d'))
    logger.info('Posting top post with permalink: {}'.format(permalink))
    logger.info(title)
    logger.info(body)
    steem.commit.post(author=account,
                      title=title,
                      body=body,
                      permlink=permalink,
                      self_vote=True,
                      tags=tfbp.TAGS)

    return permalink


def comment_on_own_top_list(sorted_post_frame, steem_or_args, account,
                            topN_permalink, Kstart=10, Kend=25):
    """ Adds the more ranks as a comment

    Parameters
    ----------
    sorted_post_frame: DataFrame
    steem_or_args: kwargs or Steem object
    account: str
    topN_permalink: str
    Kstart: int
    Kend: int

    """
    steem = tfgd.check_and_convert_steem(steem_or_args)

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
        post.reply(body=comment, author=account)
    except PostDoesNotExist:
        logger.exception('No broadcast, heh?')


def vote_and_comment_on_topK(sorted_post_frame, steem_or_args, account,
                             topN_permalink, K=25):
    """

    Parameters
    ----------
    sorted_post_frame: DataFrame
    steem_or_args: kwargs or Steem object
    account: str
    topN_permalink: str
    K: int
        number of truffles to comment and upvote

    """
    logger.info('Voting and commenting on {} top truffles'.format(K))
    steem = tfgd.check_and_convert_steem(steem_or_args)
    weight = min(840.0 / K, 100)
    topN_link = 'https://steemit.com/@{author}/{permalink}'.format(author=account,
                                                    permalink=topN_permalink)

    for kdx, (_, row) in enumerate(sorted_post_frame.iterrows()):
        if kdx >= K:
            break
        try:
            logger.info('Voting and commenting on https://steemit.com/@{author}/{permalink}'
                        ''.format(author=row.author, permalink=row.permalink))
            reply = tfbp.truffle_comment(reward=row.predicted_reward,
                                         votes=row.predicted_votes,
                                         rank=kdx + 1,
                                         topN_link=topN_link)
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


def create_wallet(steem_or_args, password, posting_key):
    """ Creates a new wallet

    Does nothing if wallet database entry already exists

    Parameters
    ----------
    steem_or_args: kwargs or Steem object
    password: str
    posting_key: str

    """
    if posting_key is None or password is None:
        raise RuntimeError('Key or password are None!')

    steem = tfgd.check_and_convert_steem(steem_or_args)

    logger.info('Unlocking or creating wallet')
    wallet = steem.wallet
    wallet.unlock(pwd=password)
    logger.info('Adding Posting Key')
    try:
        wallet.addPrivateKey(posting_key)
    except ValueError:
        logger.info('Key already present')
    logger.info('Wallet is ready')