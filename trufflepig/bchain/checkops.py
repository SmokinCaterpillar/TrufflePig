import logging
import multiprocessing as mp

from steem import Steem
from steem.post import Post

from trufflepig.utils import progressbar, error_retry, none_error_retry
import trufflepig.bchain.getdata as tpbg


logger = logging.getLogger(__name__)


def extract_comment_authors_and_permalinks(operations, account):
    """Takes a list of ops and returns a set of author and permalink tuples"""
    authors_and_permalinks = []
    for operation in operations:
        try:
            op = operation['op']
            if op[0] == 'comment':
                body = op[1]['body']
                if body.startswith('@' + account):
                    comment_author = op[1]['author']
                    comment_permalink = op[1]['permlink']

                    authors_and_permalinks.append((comment_author,
                                                   comment_permalink))
        except Exception:
            logger.exception('Could not scan operation {}'.format(operation))
    return authors_and_permalinks


def check_all_ops_in_block(block_num, steem, account):
    """ Gets all posts from one block

    Parameters
    ----------
    block_num: int
    steem: Steem
    account str

    Returns
    -------
    List of tuples with comment authors and permalinks

    """
    operations = none_error_retry(steem.get_ops_in_block)(block_num, False)
    if operations:
        return extract_comment_authors_and_permalinks(operations, account)
    else:
        logger.warning('Could not find any operations for block {}'.format(block_num))
    return []


def check_all_ops_between(start_datetime, end_datetime, steem,
                            account, stop_after=None):
    """ Queries all posts found in blocks between start and end

    Parameters
    ----------
    start_datetime: datetime
    end_datetime: datetime
    steem: Steem
    account: str
    stop_after: int or None
        For debugging

    Returns
    -------
    List of dicts of posts

    """
    start_num, block_start_datetime = tpbg.find_nearest_block_num(start_datetime, steem)
    end_num, block_end_datetime = tpbg.find_nearest_block_num(end_datetime, steem)

    total = end_num - start_num
    comment_authors_and_permalinks = []
    logger.info('Checking all operations for account {}  between '
                '{} (block {}) and {} (block {})'.format(account,
                                                         block_start_datetime,
                                                         start_num,
                                                         block_end_datetime,
                                                         end_num))

    for idx, block_num in enumerate(range(start_num, end_num+1)):
        authors_and_permalinks = check_all_ops_in_block(block_num, steem, account)
        comment_authors_and_permalinks.extend(authors_and_permalinks)
        if progressbar(idx, total, percentage_step=1, logger=logger):
            logger.info('Finished block {} '
                    '(last is {}) found so far {} '
                    'comments mentioning me...'.format(block_num, end_num,
                                                       len(comment_authors_and_permalinks)))
        if stop_after is not None and idx >= stop_after:
            break

    logger.info('Scraped {} comments mentioning me'.format(len(comment_authors_and_permalinks)))
    return comment_authors_and_permalinks


def _check_all_ops_parallel(block_nums, steem, account,
                                       stop_after=None):
    """Helper wrapper for multiprocessing"""
    comment_authors_and_permalinks = []
    for idx, block_num in enumerate(block_nums):
        authors_and_permalinks = check_all_ops_in_block(block_num, steem, account)
        comment_authors_and_permalinks.extend(authors_and_permalinks)
        if stop_after is not None and idx >= stop_after:
            break
    return comment_authors_and_permalinks


def check_all_ops_between_parallel(start_datetime, end_datetime, steem,
                                   account, stop_after=None, ncores=8,
                                   chunksize=20, timeout=1200):
    """As above but in parallel with `ncores` jobs of `chunksize`.

    Waits for comments unitl `timeout`.
    """
    start_num, block_start_datetime = tpbg.find_nearest_block_num(start_datetime, steem)
    end_num, block_end_datetime = tpbg.find_nearest_block_num(end_datetime, steem)

    logger.info('Querying IN PARALLEL with {} cores all posts between '
                '{} (block {}) and {} (block {})'.format(ncores,
                                                         block_start_datetime,
                                                         start_num,
                                                         block_end_datetime,
                                                         end_num))
    block_nums = list(range(start_num, end_num + 1))
    chunks = [block_nums[irun: irun + chunksize]
                for irun in range(0, len(block_nums), chunksize)]

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(ncores, initializer=tpbg.config_mp_logging)

    async_results = []
    for idx, chunk in enumerate(chunks):
        result = pool.apply_async(_check_all_ops_parallel,
                                  args=(chunk, steem,
                                        account,
                                        stop_after))
        async_results.append(result)
        if stop_after is not None and idx >= stop_after:
            break

    pool.close()

    comment_authors_and_permalinks = []
    for kdx, async in enumerate(async_results):
        try:
            new_posts = async.get(timeout=timeout)
            comment_authors_and_permalinks.extend(new_posts)
            if progressbar(kdx, len(chunks), percentage_step=5, logger=logger):
                logger.info('Finished chunk {} '
                            'out of {} found so far {} '
                            'comments about {}...'.format(kdx + 1, len(chunks),
                                              len(comment_authors_and_permalinks),
                                                          account))
        except TimeoutError:
            logger.exception('Something went totally wrong dude!')

    pool.join()
    return comment_authors_and_permalinks


def get_parent_posts(comment_authors_and_permalinks, steem):
    """ Scrapes posts where a reply mentioned the bot

    Parameters
    ----------
    comment_authors_and_permalinks: list of tuples
    steem: Steem object

    Returns
    -------
    List of dict
        Posts were the bot was mentioned in the replies

    """
    posts = []
    for comment_author, comment_permalink in comment_authors_and_permalinks:
        try:
            comment = error_retry(Post,
                            errors=Exception,
                            sleep_time=0.5,
                            retries=7)('@{}/{}'.format(comment_author,
                                           comment_permalink), steem)

            p = error_retry(Post,
                            errors=Exception,
                            sleep_time=0.5,
                            retries=7)(comment.root_identifier, steem)

            post = {
                'title': p.title,
                'reward': p.reward.amount,
                'votes': len([x for x in p.active_votes if x['percent'] > 0]),
                'active_votes': p.active_votes,
                'created': p.created,
                'tags': p.tags,
                'body': p.body,
                'author': p.author,
                'permalink': p.permlink,
                'comment_author': comment_author,
                'comment_permalink': comment_permalink
            }
            posts.append(post)

        except Exception as e:
            logger.exception(('Could not work with comment {} by '
                              '{}. Reconnecting...'
                              '').format(comment_permalink, comment_author))
            steem.reconnect()

    return posts
