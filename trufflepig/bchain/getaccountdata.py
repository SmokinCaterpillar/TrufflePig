import logging

import pandas as pd

from steem.account import Account


logger = logging.getLogger(__name__)


def find_nearest_index(target_datetime,
                           account,
                           steem,
                           latest_index=None,
                           max_tries=5000,
                           index_tolerance=5):
    """ Finds nearest account action index to `target_datetime`

    Currently NOT used in production!

    Parameters
    ----------
    target_datetime: datetime
    steem: Steem object
    latest_index: int
        latest index number in acount index
        leave None to get from steem directly
    max_tries: int
        number of maximum tries
    index_tolerance: int
        tolerance too closest index number

    Returns
    -------
    int: best matching index
    datetime: datetime of matching index

    """
    acc = Account(account, steem)

    if latest_index is None:
        latest_index = next(acc.history_reverse(batch_size=1))['index']

    current_index = latest_index
    best_largest_index = latest_index

    action = next(acc.get_account_history(best_largest_index, limit=10))
    best_largest_datetime = pd.to_datetime(action['timestamp'])
    if target_datetime > best_largest_datetime:
        logger.warning('Target beyond largest block num')
        return latest_index, best_largest_datetime

    best_smallest_index = 1
    increase = index_tolerance + 1
    for _ in range(max_tries):
        try:
            action = next(acc.get_account_history(current_index, limit=10))
            current_datetime = pd.to_datetime(action['timestamp'])
            if increase <= index_tolerance:
                return current_index, current_datetime
            else:

                if current_datetime < target_datetime:
                    best_smallest_index = current_index
                else:
                    best_largest_index = current_index

                increase = (best_largest_index - best_smallest_index) // 2
                current_index = best_smallest_index + increase

                if current_index < 0 or current_index > latest_index:
                    raise RuntimeError('Seriously?')
        except Exception:
            logger.exception('Problems for index {}'.format(current_index))
            current_index -= 1
            best_smallest_index -= 1
