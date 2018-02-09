import logging
from collections import OrderedDict

import pandas as pd
from steem.blockchain import Blockchain


logger = logging.getLogger(__name__)


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


def find_nearest_block_num(target_datetime, steem, latest_block_num,
                           max_tries=5000,
                           block_num_tolerance=5):
    """ Finds nearest block number to `target_datetime`

    Parameters
    ----------
    target_datetime: datetime
    steem: Steem object
    latest_block_num: int
        latest block number in bchain
    max_tries: int
        number of maximum tries
    block_num_tolerance: int
        tolerance too closest in block

    Returns
    -------
    int: best matching block number
    datetime: datetime of matching block

    """
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