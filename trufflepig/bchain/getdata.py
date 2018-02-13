import logging
import os
import multiprocessing as mp
from collections import OrderedDict

import pandas as pd
from steem import Steem
from steem.blockchain import Blockchain
from steem.post import Post, PostDoesNotExist
import json
from json import JSONDecodeError

from trufflepig.utils import progressbar


logger = logging.getLogger(__name__)

MIN_CHARACTERS = 1024

FILENAME_TEMPLATE = 'steemit_posts__{time}.gz'


def steem2bchain(steem):
    return Blockchain(steem)


def check_and_convert_steem(steem_or_args_kwargs):
    if isinstance(steem_or_args_kwargs, dict):
        return Steem(**steem_or_args_kwargs)
    elif isinstance(steem_or_args_kwargs, (list, tuple)):
        return Steem(*steem_or_args_kwargs)
    else:
        return steem_or_args_kwargs


def get_block_headers_between_offset_start(start_datetime, end_datetime,
                                           end_offset_num, steem):
    """ Returns block headers between a date range

    Parameters
    ----------
    start_datetime: datetime
    end_datetime: datetime
    end_offset_num: offset from wich to seach backwards
    steem: Steem object

    Returns
    -------
    Ordereddict: block_num -> header

    """
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    current_block_num = end_offset_num
    headers = OrderedDict()
    logger.info('Collecting header infos')
    while True:
        try:
            header = steem.get_block_header(current_block_num)
            current_datetime = pd.to_datetime(header['timestamp'])
            if start_datetime <= current_datetime and current_datetime <= end_datetime:
                header['timestamp'] = current_datetime
                headers[current_block_num] = header
            if current_datetime < start_datetime:
                break
        except Exception:
            logger.exception('Error for block num {}'.format(current_block_num))
        current_block_num -= 1
        if current_block_num % 100 == 99:
            logger.debug('Bin alread {} headers'.format(len(headers)))
    return headers


def find_nearest_block_num(target_datetime, steem,
                           latest_block_num=None,
                           max_tries=5000,
                           block_num_tolerance=5):
    """ Finds nearest block number to `target_datetime`

    Parameters
    ----------
    target_datetime: datetime
    steem: Steem object
    latest_block_num: int
        latest block number in bchain
        leave None to get from steem directly
    max_tries: int
        number of maximum tries
    block_num_tolerance: int
        tolerance too closest in block

    Returns
    -------
    int: best matching block number
    datetime: datetime of matching block

    """
    if latest_block_num is None:
        latest_block_num = steem2bchain(steem).get_current_block_num()

    current_block_num = latest_block_num
    best_largest_block_num = latest_block_num

    header = steem.get_block_header(best_largest_block_num)
    best_largest_datetime = pd.to_datetime(header['timestamp'])
    if target_datetime > best_largest_datetime:
        logger.warning('Target beyond largest block num')
        return latest_block_num, best_largest_datetime

    best_smallest_block_num = 1
    increase = block_num_tolerance + 1
    for _ in range(max_tries):
        try:
            header = steem.get_block_header(current_block_num)
            current_datetime = pd.to_datetime(header['timestamp'])
            if increase <= block_num_tolerance:
                return current_block_num, current_datetime
            else:

                if current_datetime < target_datetime:
                    best_smallest_block_num = current_block_num
                else:
                    best_largest_block_num = current_block_num

                increase = (best_largest_block_num - best_smallest_block_num) // 2
                current_block_num = best_smallest_block_num + increase

                if current_block_num < 0 or current_block_num > latest_block_num:
                    raise RuntimeError('Seriously?')
        except Exception:
            logger.exception('Problems for block num {}'.format(current_block_num))
            current_block_num -= 1
            best_smallest_block_num -= 1


def get_block_headers_between(start_datetime, end_datetime, steem):
    """ Returns block headers between two dates"""
    latest_block_num = Blockchain(steem).get_current_block_num()
    end_offset_num, _ = find_nearest_block_num(end_datetime, steem, latest_block_num)
    return get_block_headers_between_offset_start(start_datetime, end_datetime,
                                                  steem=steem,
                                                  end_offset_num=end_offset_num)


def extract_authors_and_permalinks(operations):
    authors_and_permalinks = set()
    for operation in operations:
        op = operation['op']
        if op[0] == 'comment':
            title = op[1]['title']
            body = op[1]['body']
            if title != '' and op[1]['json_metadata'] != '' and len(body) >= MIN_CHARACTERS:
                try:
                    metadata = json.loads(op[1]['json_metadata'])
                except JSONDecodeError:
                    logger.debug('Could not decode metadata for {}'.format(op))
                    continue
                try:
                    tags = metadata['tags']
                except KeyError as e:
                    logger.debug('No tags for for {}'.format(op))
                    continue
                except TypeError as e:
                    logger.debug('Type Error for for {}'.format(op))
                    continue
                try:
                    _ = tags[0]
                except IndexError as e:
                    logger.debug('Tags empty for {}'.format(op))
                    continue
                author = op[1]['author']
                permalink = op[1]['permlink']
                authors_and_permalinks.add((author, permalink))
    return list(authors_and_permalinks)


def get_post_data(authors_and_permalinks, steem):
    posts = []
    for kdx, (author, permalink) in enumerate(authors_and_permalinks):
        try:
            p = Post('@{}/{}'.format(author, permalink), steem)
        except PostDoesNotExist:
            # This happens to oftern we will suppress this
            logger.debug('Post {} by {} does not exist!'.format(author,
                                                                permalink))
            continue
        except Exception:
            logger.exception('Error in loading post {} by {}'.format(author,
                                                                     permalink))
            continue

        post = {
            'title': p.title,
            'reward': p.reward.amount,
            'votes': len(p.active_votes),
            'created': p.created,
            'tags': p.tags,
            'body': p.body,
            'author': author,
            'permalink': permalink
        }
        posts.append(post)
    return posts


def get_all_posts_from_block(block_num, steem):
    operations = steem.get_ops_in_block(block_num, False)
    if operations:
        authors_and_permalinks = extract_authors_and_permalinks(operations)
        if authors_and_permalinks:
            return get_post_data(authors_and_permalinks, steem)
        else:
            logger.debug('Could not find any posts for block {}'.format(block_num))
    else:
        logger.warning('Could not find any operations for block {}'.format(block_num))
    return []


def get_all_posts_between(start_datetime, end_datetime, steem,
                          stop_after=None):
    start_num, block_start_datetime = find_nearest_block_num(start_datetime, steem)
    end_num, block_end_datetime = find_nearest_block_num(end_datetime, steem)

    total = end_num - start_num
    posts = []
    logger.info('Querying all posts between '
                '{} (block {}) and {} (block {})'.format(block_start_datetime,
                                                         start_num,
                                                         block_end_datetime,
                                                         end_num))
    for idx, block_num in enumerate(range(start_num, end_num+1)):
        posts_in_block = get_all_posts_from_block(block_num, steem)
        posts.extend(posts_in_block)
        if progressbar(idx, total, percentage_step=1, logger=logger):
            logger.info('Finished block {} '
                    '(last is {}) found so far {} '
                    'posts...'.format(block_num, end_num, len(posts)))
        if stop_after is not None and len(posts) >= stop_after:
            break

    logger.info('Scraped {} posts'.format(len(posts)))
    return posts


def _get_all_posts_for_blocks_parallel(block_nums, steem_args,
                                       stop_after=None):
    steem = check_and_convert_steem(steem_args)
    posts = []
    for block_num in block_nums:
        posts_in_block = get_all_posts_from_block(block_num, steem)
        posts.extend(posts_in_block)
        if stop_after is not None and len(posts) >= stop_after:
            break
    return posts


def get_all_posts_between_parallel(start_datetime, end_datetime, steem_args,
                                   stop_after=None, ncores=8,
                                   chunksize=20, timeout=3600):
    steem = check_and_convert_steem(steem_args)
    start_num, block_start_datetime = find_nearest_block_num(start_datetime, steem)
    end_num, block_end_datetime = find_nearest_block_num(end_datetime, steem)

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
    pool = ctx.Pool(ncores, initializer=config_mp_logging)

    async_results = []
    for idx, chunk in enumerate(chunks):
        result = pool.apply_async(_get_all_posts_for_blocks_parallel,
                                  args=(chunk, steem_args, stop_after))
        async_results.append(result)
        if stop_after is not None and idx >= stop_after:
            break

    pool.close()

    posts = []
    for kdx, async in enumerate(async_results):
        try:
            new_posts = async.get(timeout=timeout)
            posts.extend(new_posts)
            if progressbar(kdx, len(chunks), percentage_step=5, logger=logger):
                logger.info('Finished chunk {} '
                            'out of {} found so far {} '
                            'posts...'.format(kdx + 1, len(chunks), len(posts)))
        except TimeoutError:
            logger.exception('Something went totally wrong dude!')

    pool.join()
    return posts


def scrape_or_load_full_day(date, steem_or_args, directory, overwrite=False,
                            store=True, stop_after=None, ncores=1):
    start_datetime = pd.to_datetime(date)
    end_datetime = start_datetime + pd.Timedelta(days=1)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename = FILENAME_TEMPLATE.format(time=start_datetime.strftime('%Y-%m-%d'))
    filename = os.path.join(directory,filename)
    if os.path.isfile(filename) and not overwrite:
        logger.info('Found file {} will load it'.format(filename))
        post_frame = pd.read_pickle(filename, compression='gzip')
    else:
        logger.info('File {} not found, will start scraping'.format(filename))

        if ncores == 1:
            steem = check_and_convert_steem(steem_or_args)
            posts = get_all_posts_between(start_datetime, end_datetime, steem,
                                          stop_after=stop_after)
        else:
            posts = get_all_posts_between_parallel(start_datetime, end_datetime,
                                                   steem_or_args,
                                                   stop_after=stop_after,
                                                   ncores=ncores)

        post_frame = pd.DataFrame(data=posts, columns=sorted(posts[0].keys()))
        if store:
            logger.info('Storing file {} to disk'.format(filename))
            post_frame.to_pickle(filename, compression='gzip')
    return post_frame


def config_mp_logging(level=logging.INFO):
    logging.basicConfig(level=level)


def scrape_or_load_training_data(steem_or_args, directory,
                                 days=20, offset_days=8,
                                 ncores=8,
                                 current_datetime=None,
                                 stop_after=None):

    if current_datetime is None:
        current_datetime = pd.datetime.utcnow()
    else:
        current_datetime = pd.to_datetime(current_datetime)

    start_datetime = current_datetime - pd.Timedelta(days=days + offset_days)

    frames = []
    for day in range(days):
        next_date = (start_datetime + pd.Timedelta(days=day)).date()
        frame = scrape_or_load_full_day(next_date, steem_or_args,
                                        directory, overwrite=False,
                                        store=True, stop_after=stop_after,
                                        ncores=ncores)
        frames.append(frame)
    return pd.concat(frames, axis=0)


def scrape_hour_data(steem_or_args, hours=25,
                     offset_hours=24,
                     current_datetime=None,
                     ncores=8, stop_after=None):
    if current_datetime is None:
        current_datetime = pd.datetime.utcnow()
    else:
        current_datetime = pd.to_datetime(current_datetime)

    end_datetime = current_datetime - pd.Timedelta(hours=offset_hours)
    start_datetime = end_datetime - pd.Timedelta(hours=hours)

    if ncores == 1:
        steem = check_and_convert_steem(steem_or_args)
        posts = get_all_posts_between(start_datetime, end_datetime,
                                      steem, stop_after=stop_after)
    else:
        posts = get_all_posts_between_parallel(start_datetime,
                                               end_datetime,
                                               steem_or_args,
                                               stop_after=stop_after,
                                               ncores=ncores)

    post_frame = pd.DataFrame(data=posts, columns=sorted(posts[0].keys()))
    return post_frame
