import logging
import multiprocessing as mp
import time

import pandas as pd
import numpy as np
from steem.account import Account

import trufflepig.bchain.getdata as tpbg
from trufflepig.utils import progressbar


logger = logging.getLogger(__name__)


MEMO_START = 'https://steemit.com/'

BITBOTS = list({'smartmarket', 'smartsteem', 'upme', 'randowhale',
            'minnowbooster', 'boomerang', 'booster', 'hak4life',
            'lays', 'speedvoter', 'ebargains', 'danzy', 'bumper',
            'upvotewhale', 'treeplanter', 'minnowpond', 'morwhale',
            'drotto', 'postdoctor', 'moonbot', 'tipu', 'blockgators',
            'echowhale', 'steemvote', 'byresteem', 'originalworks', 'withsmn',
            'siditech', 'alphaprime', 'hugewhale', 'steemvoter', 'hottopic',
            'resteemable', 'earthnation-bot', 'photocontests', 'friends-bot',
           'followforupvotes', 'frontrunner', 'resteembot', 'steemlike',
           'thundercurator', 'earnmoresteem', 'microbot', 'coolbot',
           'thehumanbot', 'steemthat', 'gangvote', 'refresh', 'cabbage-dealer',
           'growingpower', 'postresteem', 'mecurator', 'talhadogan',
           'okankarol', 'bidseption', 'highvote', 'oguzhangazi', 'ottoman',
           'resteemr', 'superbot', 'bestvote', 'zerotoherobot', 'red-rose',
           'jeryalex', 'oceansbot', 'fresteem', 'otobot', 'bidbot',
           'honestbot', 'upgoater', 'whalebuilder', 'postpromoter', 'pwrup',
           'spydo', 'upmewhale', 'promobot', 'puppybot', 'moneymatchgaming',
           'sneaky-ninja', 'zapzap', 'sleeplesswhale', 'estream.studios',
           'seakraken', 'canalcrypto', 'upmyvote', 'hotbot',
           'redlambo', 'slimwhale', 'singing.beauty', 'inciter', 'lovejuice',
           'steembidbot', 'bid4joy', 'mitsuko', 'pushup', 'luckyvotes',
           'discordia', 'shares', 'postdoctor', 'upboater',
           'megabot', 'dailyupvotes', 'ebargains', 'bluebot', 'upyou',
           'edensgarden', 'smartwhale', 'voterunner', 'nado.bot',
           'jerrybanfield', 'foxyd', 'onlyprofitbot', 'minnowhelper',
           'msp-bidbot', 'therising', 'bearwards', 'thebot', 'buildawhale',
           'chronocrypto', 'brupvoter', 'smartsteem', 'payforplay',
           'adriatik', 'cryptoempire', 'isotonic', 'minnowfairy',
           'appreciator', 'childfund', 'mercurybot', 'allaz', 'sunrawhale',
           'mrswhale', 'kittybot', 'lightningbolt', 'hottopic',
           'sportic'})


def find_nearest_index(target_datetime,
                           account,
                           steem,
                           latest_index=None,
                           max_tries=5000,
                           index_tolerance=5):
    """ Finds nearest account action index to `target_datetime`

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

    action = next(acc.get_account_history(best_largest_index, limit=1))
    best_largest_datetime = pd.to_datetime(action['timestamp'])
    if target_datetime > best_largest_datetime:
        logger.debug('Target beyond largest block num')
        return latest_index, best_largest_datetime

    best_smallest_index = 1
    increase = index_tolerance + 1
    for _ in range(max_tries):
        try:
            action = next(acc.get_account_history(current_index, limit=1))
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
                    raise RuntimeError('Seriously? Error for '
                                       'account {}: current_index {} '
                                       'latest_index {}'.format(account,
                                                                current_index,
                                                                latest_index))
        except Exception:
            logger.exception('Problems for index {}'.format(current_index))
            current_index += 1
            best_largest_index += 1


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


def get_upvote_payments(account, steem, min_datetime, max_datetime,
                        batch_size=1000, max_time=1800):

    start = time.time()
    upvote_payments = {}

    start_index, _ = find_nearest_index(max_datetime,
                                     account, steem)
    try:
        transfers = history_reverse(account, steem, filter_by='transfer',
                                    start_index=start_index,
                                    batch_size=batch_size)
    except Exception as e:
        logger.exception('Could not get account data from {}'.format(account))
        transfers = []

    for transfer in transfers:
        try:
            memo = transfer['memo']
            if memo.startswith(MEMO_START):
                author, permalink = memo.split('/')[-2:]
                if author.startswith('@'):
                    author = author[1:]
                    if (author, permalink) not in upvote_payments:
                        upvote_payments[(author, permalink)] = {}
                    trx_id = transfer['trx_id']
                    amount = transfer['amount']
                    upvote_payments[(author, permalink)][trx_id] = amount

            timestamp = pd.to_datetime(transfer['timestamp'])
            if timestamp < min_datetime:
                break

            now = time.time()
            if now - start > max_time:
                logger.error('Reached max time of {} seconds '
                             ' will stop! Account {} from {} until {} '
                             'last timestamp {}'.format(max_time,
                                                        account,
                                                        min_datetime,
                                                        max_datetime,
                                                        timestamp))
                break

        except Exception as e:
            logger.exception('Could not parse {}'.format(transfer))

    return upvote_payments


def history_reverse(account, steem, start_index, filter_by=None,
                    batch_size=1000, raw_output=False):
        """ Stream account history in reverse chronological order."""
        acc = Account(account, steem)
        i = start_index
        if batch_size > start_index:
            batch_size = start_index
        while i > 0:
            if i - batch_size < 0:
                batch_size = i
            yield from acc.get_account_history(
                index=i,
                limit=batch_size,
                order=-1,
                filter_by=filter_by,
                raw_output=raw_output,
            )
            i -= (batch_size + 1)


def extend_upvotes_and_payments(upvote_payments, new_payments):
    for author_permalink, new_upvotes in new_payments.items():
            if author_permalink not in upvote_payments:
                upvote_payments[author_permalink] = {}
            upvote_payments[author_permalink].update(new_upvotes)
    return upvote_payments


def _get_upvote_payments_parrallel(accounts, steem, min_datetime,
                                   max_datetime):
    results = {}
    for account in accounts:
        result = get_upvote_payments(account, steem, min_datetime, max_datetime)
        results = extend_upvotes_and_payments(results, result)

    return result


def get_upvote_payments_for_accounts(accounts, steem, min_datetime, max_datetime,
                                     chunksize=10, ncores=20, timeout=3600):
    logger.info('Querying upvote purchases between {} and '
                '{} for {} accounts'.format(min_datetime,
                                            max_datetime,
                                            len(accounts)))

    # do queries by day!
    start_datetimes = pd.date_range(min_datetime, max_datetime).tolist()
    end_datetimes = [x for x in start_datetimes[1:]] + [max_datetime]

    if ncores > 1:
        chunks = [accounts[irun: irun + chunksize]
                  for irun in range(0, len(accounts), chunksize)]

        ctx = mp.get_context('spawn')
        pool = ctx.Pool(ncores, initializer=tpbg.config_mp_logging)

        async_results = []
        for start_datetime, end_datetime in zip(start_datetimes, end_datetimes):
            for idx, chunk in enumerate(chunks):
                result = pool.apply_async(_get_upvote_payments_parrallel,
                                          args=(chunk, steem,
                                                start_datetime, end_datetime))
                async_results.append(result)

        pool.close()

        upvote_payments = {}
        for kdx, async in enumerate(async_results):
            try:
                payments = async.get(timeout=timeout)
                upvote_payments = extend_upvotes_and_payments(upvote_payments,
                                                              payments)
                if progressbar(kdx, len(async_results), percentage_step=5, logger=logger):
                    logger.info('Finished chunk {} '
                                'out of {} found so far {} '
                                'upvote buyers...'.format(kdx + 1, len(async_results), len(upvote_payments)))
            except Exception as e:
                logger.exception('Something went totally wrong dude!')

        pool.join()
    else:
        return _get_upvote_payments_parrallel(accounts, steem, min_datetime,
                                              max_datetime)

    logger.info('Found {} upvote bought articles'.format(len(upvote_payments)))
    return upvote_payments


def get_upvote_payments_to_bots(steem, min_datetime, max_datetime,
                                bots=BITBOTS, ncores=30):
    logger.info('Getting payments to following bots {}'.format(bots))
    return get_upvote_payments_for_accounts(accounts=bots,
                                            steem=steem,
                                            min_datetime=min_datetime,
                                            max_datetime=max_datetime,
                                            ncores=ncores,
                                            chunksize=1)
