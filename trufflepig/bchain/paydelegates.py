import logging

import trufflepig.bchain.getaccountdata as tpga
import trufflepig.bchain.getdata as tpdg

from steem import Steem
from steembase import operations
from steem.account import Account


logger = logging.getLogger(__name__)


INVESTOR_SHARE = 0.5

MEMO = 'Thank you for your trust in TrufflePig the Artificial Intelligence bot to help content curators and minnows.'


def pay_delegates(account, steem_args, investor_share=INVESTOR_SHARE,
                  memo=MEMO):
    """ Pays delegators their share of daily SBD rewards

    Parameters
    ----------
    account: str
    steem_args: Steem or kwargs
    investor_share: float
    memo: str

    """
    logger.info('Computing payouts for delegates!')
    steem = tpdg.check_and_convert_steem(steem_args)
    payouts = tpga.get_delegate_payouts(account, steem,
                                        investor_share=investor_share)
    logger.info('Count the following payouts:\n{}'.format(payouts))
    claim_all_reward_balance(steem, account)
    for delegator, payout in payouts.items():
        try:
            if payout:
                logger.info('Paying {} SBD to {}'.format(delegator, payout))
                steem.commit.transfer(to=delegator,
                                      amount=payout,
                                      asset='SBD',
                                      memo=memo,
                                      account=account)
        except Exception as e:
            logger.exception('Could not pay {} SBD to {}!'.format(payout,
                                                                  delegator))


def claim_all_reward_balance(steem, account):
    """Helper funtion to claim rewards because of bug in Steem object"""
    acc = Account(account, steem)
    reward_steem = acc['reward_steem_balance']
    reward_sbd = acc['reward_sbd_balance']
    reward_vests = acc['reward_vesting_balance']
    op = operations.ClaimRewardBalance(
            account=account,
            reward_steem=reward_steem,
            reward_sbd=reward_sbd,
            reward_vests=reward_vests,
        )
    return steem.commit.finalizeOp(op, account, "posting")
