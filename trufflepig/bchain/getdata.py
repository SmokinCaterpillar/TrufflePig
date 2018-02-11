import logging
import os
from collections import OrderedDict

import pandas as pd
from steem.blockchain import Blockchain
from steem.post import Post
import json
from json import JSONDecodeError

from trufflepig.utils import progressbar


logger = logging.getLogger(__name__)

MIN_CHARACTERS = 1024

FILENAME_TEMPLATE = 'steemit_posts__{year:04d}-{month:02d}-{day:02d}.pkl'


def steem2bchain(steem):
    return Blockchain(steem)


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
    authors_and_permalinks = []
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
                authors_and_permalinks.append((author, permalink))
    return authors_and_permalinks


def get_post_data(authors_and_permalinks, steem):
    posts = []
    for kdx, (author, permalink) in enumerate(authors_and_permalinks):
        try:
            p = Post('@{}/{}'.format(author, permalink), steem)
        except Exception as e:
            print(repr(e))
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
    start_num, _ = find_nearest_block_num(start_datetime, steem)
    end_num, _ = find_nearest_block_num(end_datetime, steem)

    total = end_num - start_num
    posts = []
    logger.info('Querying all posts between '
                '{} (block {}) and {} (block {})'.format(start_datetime,
                                                         start_num,
                                                         end_datetime,
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


def scrape_or_load_full_day(date, steem, directory, overwrite=False,
                            store=True,
                            stop_after=None):
    start_datetime = pd.to_datetime(date)
    end_datetime = start_datetime + pd.Timedelta(days=1)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename = FILENAME_TEMPLATE.format(year=start_datetime.year,
                                        month=start_datetime.month,
                                        day=start_datetime.day)
    filename = os.path.join(directory,filename)
    if os.path.isfile(filename) and not overwrite:
        logger.info('Found file {} will load it'.format(filename))
        post_frame = pd.read_pickle(filename)
    else:
        posts = get_all_posts_between(start_datetime, end_datetime, steem,
                                      stop_after=stop_after)
        post_frame = pd.DataFrame(data=posts, columns=sorted(posts[0].keys()))
        if store:
            post_frame.to_pickle(filename, compression='gzip')
    return post_frame
