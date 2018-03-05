import logging

import pandas as pd
import numpy as np

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


def get_delegates_and_shares(account, steem):
    """ Queries all delegators to `account` and the amount of shares

    Parameters
    ----------
    account: str
    steem: Steem

    Returns
    -------
    dict of float

    """
    acc = Account(account, steem)
    delegators = {}
    for tr in acc.history_reverse(filter_by='delegate_vesting_shares'):
        try:
            delegator = tr['delegator']
            if delegator not in delegators:
                shares = tr['vesting_shares']
                if shares.endswith(' VESTS'):
                    shares = float(shares[:-6])
                    timestamp = pd.to_datetime(tr['timestamp'])
                    delegators[delegator] = {'vests': shares,
                                             'timestamp': timestamp}
                else:
                    raise RuntimeError('Weird shares {}'.format(shares))

        except Exception as e:
            logger.exception('Error extracting delegator from {}'.format(tr))
    return delegators


def get_delegate_payouts(account, steem, current_datetime,
                         min_days, investor_share):
    """ Returns pending payouts for investors

    Parameters
    ----------
    account: str
    steem: Steem
    current_datetime: datetime
    min_days: int
        minimum days of delegation before payout
    investor_share: float

    Returns
    -------
    dict of float:
        SBD to pay to each investor

    """
    assert 0 < investor_share <= 1

    current_datetime = pd.to_datetime(current_datetime)
    threshold_date = current_datetime - pd.Timedelta(days=min_days)

    vests_by = get_delegates_and_shares(account, steem)
    filtered_vests_by = {delegator: dict_['vests']
                         for delegator, dict_ in vests_by.items()
                            if dict_['timestamp'] < threshold_date}
    acc = Account(account, steem)

    pending = acc.balances['rewards']['SBD']
    vests = acc.balances['total']['VESTS']
    filtered_vests_by[account] = vests

    total_vests = sum(filtered_vests_by.values())
    payouts = {delegator: np.round(vests / total_vests * investor_share * pending, decimals=3)
                    for delegator, vests in filtered_vests_by.items() if delegator != account}

    return payouts

