import logging

from steem.post import Post, PostDoesNotExist

import trufflepig.bchain.posts as tfbp
import trufflepig.bchain.getdata as tfgd


logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'daily-truffle-picks-{date}'



def post_topN_list(sorted_post_frame, steem_or_args, account, current_datetime, N=10):
    steem = tfgd.check_and_convert_steem(steem_or_args)

    df = sorted_post_frame.iloc[:N, :]

    logger.info('Creating top {} post'.format(N))
    title, body = tfbp.topN_post(topN_authors=df.author,
                             topN_permalinks=df.permalink,
                             topN_titles=df.title,
                             topN_filtered_bodies=df.filtered_body,
                             topN_votes=df.votes,
                             topN_rewards=df.reward,
                             title_date=current_datetime)

    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('Y-%m-%d'))
    logger.info('Posting top post with permalink: {}'.format(permalink))
    steem.commit.post(author=account,
                      title=title,
                      body=body,
                      permlink=permalink,
                      self_vote=True,
                      tags=tfbp.TAGS)

    return permalink


def vote_and_comment_on_topK(sorted_post_frame, steem_or_args, account, topN_permalink,
                             K=20):

    logger.info('Voting and commenting on {} top truffles'.format(K))
    steem = tfgd.check_and_convert_steem(steem_or_args)
    weight = 100.0 / K
    topN_link = 'https://steemit.com/@{author}/{permalink}'.format(author=account,
                                                    permalink=topN_permalink)

    for kdx, (_, row) in enumerate(sorted_post_frame.iterrows()):
        if kdx >= K:
            break
        try:
            logger.info('Voting and commenting on https://steemit.com/@{author}/{permalink}'
                        ''.format(author=row.author, permalink=row.permalink))
            reply = tfbp.truffle_comment(reward=row.reward,
                                         votes=row.votes,
                                         rank=kdx + 1,
                                         topN_link=topN_link)
            post = Post('@{}/{}'.format(row.author, row.permalink), steem)
            post.upvote(weight=weight, voter=account)
            post.reply(body=reply, author=account)
        except PostDoesNotExist:
            logger.exception('Post not found of row {}'.format(row))