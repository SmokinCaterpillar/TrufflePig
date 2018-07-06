from steem.amount import Amount
import trufflepig.bchain.postdata as tbpd
import logging
import numpy as np


logger = logging.getLogger(__name__)



def compute_total_sbd(upvote_payments):
    sbd = 0
    steem = 0
    for (author, permalink), payments in upvote_payments.items():
        for payment in payments.values():
            amount = Amount(payment['amount'])
            value = amount.amount
            asset = amount.asset
            if asset == 'SBD':
                sbd += value
            elif asset == 'STEEM':
                steem += value
    return sbd, steem


def create_trending_post(post_frame, upvote_payments, poster, topN_permalink,
                         overview_permalink, current_datetime, bots=()):
    total_paid_sbd, total_paid_steem = compute_total_sbd(upvote_payments)

    bots = set(bots)

    logger.info('People spend {} SBD and {} Steem on Bid Bots the last 24 '
                'hours.'.format(total_paid_sbd, total_paid_steem))

    # exclude bit bots
    no_bid_bots_frame = post_frame.loc[post_frame.bought_votes == 0, :].copy()

    # exclude self votes
    self_votes = []
    for idx, row in no_bid_bots_frame.iterrows():
        self_votes.append(row.author in {x['voter'] for x in row.active_votes})
    self_votes = np.array(self_votes)
    no_bid_bots_frame = no_bid_bots_frame.loc[~self_votes, :]

    # exlude all bot votes
    bot_votes = []
    for idx, row in no_bid_bots_frame.iterrows():
        bot_votes.append(len(bots.intersection(
            {x['voter'] for x in row.active_votes})
        ) > 0)
    bot_votes = np.array(bot_votes)
    no_bid_bots_frame = no_bid_bots_frame.loc[~bot_votes, :]

    no_bid_bots_frame.sort_values('reward', inplace=True, ascending=False)

    logger.info('Self/Bot Voted Posts {} out of '
                '{}'.format(len(post_frame) - len(no_bid_bots_frame),
                                                  len(post_frame)))

    logger.info('TOPLIST NO BID-BOTS AND SELF-VOTES')
    for x in range(10):
        what = no_bid_bots_frame.iloc[x]
        logger.info('{rank}. [{title}](https://steemit.com/@{author}/{permalink})  --  '
                    '**by @{author} with a current reward of {reward} '
                    'SBD'.format(rank=x+1,
                                 title=what.title,
                                 author=what.author,
                                 permalink=what.permalink,
                                 reward=what.reward))

    tbpd.post_top_trending_list(no_bid_bots_frame, poster, current_datetime,
                                overview_permalink=overview_permalink,
                                trufflepicks_permalink=topN_permalink,
                                steem_amount=total_paid_steem,
                                sbd_amount=total_paid_sbd)

