import logging
import time

import pandas as pd
from steem.post import Post, PostDoesNotExist
from steem.converter import Converter

import trufflepig.model as tpmo
import trufflepig.bchain.posts as tpbp
import trufflepig.bchain.getdata as tppd
import trufflepig.bchain.getaccountdata as tpaa
from trufflepig.utils import error_retry
from trufflepig.bchain.poster import Poster

logger = logging.getLogger(__name__)


PERMALINK_TEMPLATE = 'weekly-truffle-updates-{date}'

SPELLING_CATEGORY = ['num_spelling_errors', 'errors_per_word']

STYLE_CATEGORY = [x for x in tpmo.FEATURES if x not in SPELLING_CATEGORY]

TAGS = ['steemit', 'steemstem', 'minnowsupport', 'technology', 'utopian-io']


def compute_weekly_statistics(post_frame, pipeline, N=10, topics_step=4):
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
    top_tags_earnings = top_tags.copy()
    top_tags = top_tags.iloc[:N, :]

    logger.info('Computing top tags earnings...')
    top_tags_earnings['per_post'] = top_tags_earnings.reward / top_tags_earnings['count']
    min_count = 500
    top_tags_earnings = top_tags_earnings[top_tags_earnings['count']
                                          >= min_count].sort_values('per_post', ascending=False)
    top_tags_earnings = top_tags_earnings.iloc[:N, :]

    logger.info('Computing bid bot stats...')
    num_articles = (post_frame.bought_votes > 0).sum()
    bid_bots_percent = num_articles / len(post_frame) * 100
    bid_bots_steem = post_frame.steem_bought_reward.sum()
    bid_bots_sbd = post_frame.sbd_bought_reward.sum()

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

    logger.info('Computing top tfidf...')
    topic_model = pipeline.named_steps['feature_generation'].transformer_list[1][1]
    tfidf = topic_model.tfidf
    dictionary = topic_model.dictionary
    sample_size = 2000
    if sample_size > len(post_frame):
        sample_frame = post_frame
    else:
        sample_frame = post_frame.sample(n=sample_size)
    corpus_tfidf = tfidf[topic_model.to_corpus(sample_frame.tokens)]
    top_tfidf = {}
    for doc in corpus_tfidf:
        for iWord, tf_idf in doc:
            iWord = dictionary.get(iWord)
            if iWord not in top_tfidf:
                top_tfidf[iWord] = 0
            if tf_idf > top_tfidf[iWord]:
                top_tfidf[iWord] = tf_idf
    top_tfidf = pd.Series(top_tfidf, name='score')
    top_tfidf = top_tfidf.sort_values(ascending=False).iloc[:N]

    # get top authors
    logger.info('Computing top posts...')
    top_posts = post_frame.loc[:, ['title', 'author', 'permalink', 'reward', 'votes']].sort_values('reward',
                                                                                 ascending=False).iloc[:N, :]

    # get topics
    logger.info('Computing topics...')
    num_topics = topic_model.num_topics
    topics = topic_model.print_topics(n_best=num_topics, n_words=4,
                                      topics_step=topics_step)

    # get feature importances
    logger.info('Computing feature importances...')
    feature_selector = pipeline.named_steps['feature_generation'].transformer_list[0][1]
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
          bid_bots_sbd=bid_bots_sbd,
          bid_bots_steem=bid_bots_steem,
          bid_bots_percent=bid_bots_percent,
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
          top_tags_earnings=top_tags_earnings.index,
          top_tags_earnings_counts=top_tags_earnings['count'],
          top_tags_earnings_reward=top_tags_earnings.per_post,
          top_words=top_words.index,
          top_words_counts=top_words,
          top_tfidf=top_tfidf.index,
          top_tfidf_scores=top_tfidf,
          spelling_percent=spelling_percent,
          style_percent=style_percent,
          topic_percent=topic_percent,
          topics=topics
    )
    logger.info('Done final dict:\n{}'.format(result))
    return result


def return_overview_permalink_if_exists(account, steem, current_datetime):
    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%U'))
    try:
        error_retry(Post, retries=10,
                    sleep_time=4)('@{}/{}'.format(account, permalink), steem)
        return permalink
    except PostDoesNotExist:
        return ''


def post_weakly_update(pipeline, post_frame, poster, current_datetime):
    steem_per_mvests = Converter(poster.steem).steem_per_mvests()
    stats = compute_weekly_statistics(post_frame, pipeline)

    delegator_list = tpaa.get_delegates_and_shares(poster.account, poster.steem).keys()

    title, body = tpbp.weekly_update(steem_per_mvests=steem_per_mvests,
                                     current_datetime=current_datetime,
                                     delegator_list=delegator_list,
                                     **stats)
    permalink = PERMALINK_TEMPLATE.format(date=current_datetime.strftime('%Y-%V'))

    poster.post(body=body,
                title=title,
                permalink=permalink,
                self_vote=True,
                tags=TAGS)

    return permalink
