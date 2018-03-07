import logging
import time

import pandas as pd
from steem.post import Post, PostDoesNotExist
from steem.converter import Converter

import trufflepig.model as tpmo
import trufflepig.bchain.posts as tpbp
import trufflepig.bchain.getdata as tppd


logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'weekly-truffle-updates-{date}'

SPELLING_CATEGORY = ['num_spelling_errors', 'errors_per_word']

STYLE_CATEGORY = [x for x in tpmo.FEATURES if x not in SPELLING_CATEGORY]

TAGS = ['steemit', 'steemstem', 'minnowsupport', 'technology', 'utopian-io']


def compute_weekly_statistics(post_frame, pipeline, N=10, n_topics=20):
    logger.info('Computing statistics...')
    total_reward = post_frame.reward.sum()
    total_posts = len(post_frame)
    total_votes = post_frame.votes.sum()
    start_datetime = post_frame.created.min()
    end_datetime = post_frame.created.max()
    mean_reward = post_frame.reward.mean()
    median_reward = post_frame.reward.median()
    dollar_percent = (post_frame.reward < 1).sum() / len(post_frame) * 100

    # get top tags
    logger.info('Computing top tags...')
    tag_count_dict = {}
    tag_payout = {}
    for tags, reward in zip(post_frame.tags, post_frame.reward):
        for tag in tags:
            if tag not in tag_count_dict:
                tag_count_dict[tag] = 0
                tag_payout[tag] = 0
            tag_count_dict[tag] += 1
            tag_payout[tag] += reward
    counts = pd.Series(tag_count_dict, name='count')
    rewards = pd.Series(tag_payout, name='reward')
    top_tags = counts.to_frame().join(rewards).sort_values('count',
                                                          ascending=False)
    top_tags = top_tags.iloc[:N, :]

    # get top tokens
    logger.info('Computing top words...')
    token_count_dict = {}
    for tokens in post_frame.tokens:
        for token in tokens:
            if token not in token_count_dict:
                token_count_dict[token] = 0
            token_count_dict[token] += 1
    top_words = pd.Series(token_count_dict, name='count')
    top_words = top_words.sort_values(ascending=False).iloc[:N]

    # get top authors
    logger.info('Computing top posts...')
    top_posts = post_frame.loc[:, ['title', 'author', 'permalink', 'reward', 'votes']].sort_values('reward',
                                                                                 ascending=False).iloc[:N, :]

    # get topics
    logger.info('Computing topics...')
    topic_model = pipeline.named_steps['feature_generation'].transformer_list[1][1]
    topics = topic_model.print_topics(n_best=n_topics, n_words=4)

    # get feature importances
    logger.info('Computing feature importances...')
    feature_selector = pipeline.named_steps['feature_generation'].transformer_list[0][1]
    num_topics = topic_model.num_topics
    features = feature_selector.features
    feature_names = features + ['topic_{:03d}'.format(x)
                                for x in range(num_topics)]
    spelling_percent = 0
    style_percent = 0
    topic_percent = 0
    for kdx, importance in enumerate(pipeline.named_steps['regressor'].feature_importances_):
        name = feature_names[kdx]
        importance *= 100
        if name in SPELLING_CATEGORY:
            spelling_percent += importance
        elif name in STYLE_CATEGORY:
            style_percent += importance
        else:
            topic_percent += importance

    result = dict(
          start_datetime=start_datetime,
          end_datetime=end_datetime,
          total_posts=total_posts,
          total_votes=total_votes,
          total_reward=total_reward,
          median_reward=median_reward,
          mean_reward=mean_reward,
          dollar_percent=dollar_percent,
          top_posts_authors=top_posts.author,
          top_posts_titles=top_posts.title,
          top_posts_rewards=top_posts.reward,
          top_posts_permalinks=top_posts.permalink,
          top_tags=top_tags.index,
          top_tag_counts=top_tags['count'],
          top_tag_rewards=top_tags.reward,
          top_words=top_words.index,
          top_words_counts=top_words,
          spelling_percent=spelling_percent,
          style_percent=style_percent,
          topic_percent=topic_percent,
          topics=topics
    )
    logger.info('Done final dict:\n{}'.format(result))
    return result


def return_overview_permalink_if_exists(account, steem_args, current_datetime):
    steem = tppd.check_and_convert_steem(steem_args)
    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%V'))
    try:
        Post('@{}/{}'.format(account, permalink), steem)
        return permalink
    except PostDoesNotExist:
        return ''


def post_weakly_update(pipeline, post_frame, account, steem_args, current_datetime):
    steem = tppd.check_and_convert_steem(steem_args)
    steem_per_mvests = Converter(steem).steem_per_mvests()
    stats = compute_weekly_statistics(post_frame, pipeline)

    title, body = tpbp.weekly_update(steem_per_mvests=steem_per_mvests,
                                     current_datetime=current_datetime,
                                     **stats)
    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%V'))
    logger.info('Posting weekly update with permalink: {}'.format(permalink))
    logger.info(title)
    logger.info(body)
    steem.commit.post(author=account,
                      title=title,
                      body=body,
                      permlink=permalink,
                      self_vote=True,
                      tags=TAGS)

    return permalink
