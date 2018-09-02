import logging

import trufflepig.bchain.getaccountdata as tpga
import trufflepig.bchain.getdata as tpdg
from trufflepig.utils import error_retry

from steem import Steem
from steembase import operations
from steem.account import Account
from steem.amount import Amount
from steembase.exceptions import RPCError


logger = logging.getLogger(__name__)


INVESTOR_SHARE = 0.5

MEMO = 'Thank you for your trust in TrufflePig the Artificial Intelligence bot to help content curators and minnows.'


def pay_delegates(account, steem,
                  current_datetime,
                  min_days=3,
                  investor_share=INVESTOR_SHARE,
                  memo=MEMO):
    """ Pays delegators their share of daily SBD rewards

    Parameters
    ----------
    account: str
    steem: Steem or kwargs
    current_datetime: dateime
    min_days: int
    investor_share: float
    memo: str

    """
    logger.info('Computing payouts for delegates!')
    sbd_payouts, steem_payouts = error_retry(tpga.get_delegate_payouts)(
        account, steem,
        current_datetime=current_datetime,
        min_days=min_days,
        investor_share=investor_share
    )

    claim_all_reward_balance(steem, account)

    logger.info('Found the following SBD payouts:\n{}'.format(sbd_payouts))
    for delegator, payout in sbd_payouts.items():
        try:
            if payout:
                logger.info('Paying {} SBD to {}'.format(delegator, payout))
                error_retry(steem.commit.transfer,
                            errors=(RPCError, TypeError))(to=delegator,
                                                         amount=payout,
                                                         asset='SBD',
                                                         memo=memo,
                                                         account=account)
        except Exception as e:
            logger.exception('Could not pay {} SBD to {}! '
                             'Reconnecting...'.format(payout, delegator))
            steem.reconnect()

    logger.info('Found the following STEEM payouts:\n{}'.format(steem_payouts))
    for delegator, payout in steem_payouts.items():
        try:
            if payout:
                logger.info('Paying {} STEEM to {}'.format(delegator, payout))
                error_retry(steem.commit.transfer,
                            errors=(RPCError, TypeError))(to=delegator,
                                                         amount=payout,
                                                         asset='STEEM',
                                                         memo=memo,
                                                         account=account)
        except Exception as e:
            logger.exception('Could not pay {} STEEM to {}! '
                             'Reconnecting...'.format(payout, delegator))
            steem.reconnect()


def claim_all_reward_balance(steem, account):
    """Helper funtion to claim rewards because of bug in Steem"""
    acc = Account(account, steem)
    reward_steem = acc['reward_steem_balance']
    reward_sbd = acc['reward_sbd_balance']
    reward_vests = acc['reward_vesting_balance']
    logger.info('Claiming {}, {}, and {} for {}'.format(reward_sbd,
                                                        reward_vests,
                                                        reward_steem,
                                                        account))
    op = operations.ClaimRewardBalance(account=account,
                                       reward_steem=reward_steem,
                                       reward_sbd=reward_sbd,
                                       reward_vests=reward_vests)

    can_claim = any(Amount(x).amount > 0 for x in (reward_sbd, reward_vests, reward_steem))
    if can_claim:
        try:
            return error_retry(steem.commit.finalizeOp)(op, account, "posting")
        except Exception:
            logger.exception('Could not claim rewards {}. '
                             'Reconnecting...'.format((reward_sbd,
                                                      reward_vests,
                                                      reward_steem)))
            steem.reconnect()
    else:
        logger.info('Nothing to claim!')
