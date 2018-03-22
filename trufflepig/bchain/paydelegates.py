import logging

import trufflepig.bchain.getaccountdata as tpga
import trufflepig.bchain.getdata as tpdg
from trufflepig.utils import error_retry

from steem import Steem
from steembase import operations
from steem.account import Account
from steem.amount import Amount


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
    payouts = tpga.get_delegate_payouts(account, steem,
                                        current_datetime=current_datetime,
                                        min_days=min_days,
                                        investor_share=investor_share)
    logger.info('Found the following payouts:\n{}'.format(payouts))
    claim_all_reward_balance(steem, account)
    for delegator, payout in payouts.items():
        try:
            if payout:
                logger.info('Paying {} SBD to {}'.format(delegator, payout))
                error_retry(steem.commit.transfer)(to=delegator,
                                                   amount=payout,
                                                   asset='SBD',
                                                   memo=memo,
                                                   account=account)
        except Exception as e:
            logger.exception('Could not pay {} SBD to {}!'.format(payout,
                                                                  delegator))


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
            logger.exception('Could not claim rewards {}'.format((reward_sbd,
                                                                  reward_vests,
                                                                  reward_steem)))
    else:
        logger.info('Nothing to claim!')
