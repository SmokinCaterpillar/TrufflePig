import pandas as pd

import trufflepig.bchain.posts as tbpo
import trufflepig.filters.textfilters as tptf
import trufflepig.preprocessing as tppp
from trufflepig.testutils import random_data


def test_comment():
    post = random_data.create_n_random_posts(1)[0]

    result = tbpo.truffle_comment(reward=post['reward'],
                                  votes=post['votes'],
                                  rank=1,
                                  topN_link='www.example.com',
                                  truffle_link='www.tf.tf')

    assert result


def test_topN_post():
    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()
    df['image_urls'] = df.body.apply(lambda x: tptf.get_image_urls(x))

    title, post = tbpo.topN_post(topN_authors=df.author,
                                 topN_permalinks=df.permalink,
                                 topN_titles=df.title,
                                 topN_filtered_bodies=df.filtered_body,
                                 topN_image_urls=df.image_urls,
                                 topN_rewards=df.reward, topN_votes=df.votes,
                                 title_date=date, truffle_link='de.de.de')

    assert post
    assert title


def test_topN_comment():
    posts = random_data.create_n_random_posts(25)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    post = tbpo.topN_comment(topN_authors=df.author,
                             topN_permalinks=df.permalink,
                             topN_titles=df.title,
                             topN_votes=df.votes,
                             topN_rewards=df.reward)

    assert post


def test_post_on_call():

    comment = tbpo.on_call_comment(reward=1000000, author='Douglas Adams', votes=42000000,
                                   topN_link='www.deep.thought',
                                   truffle_link='adsadsad.de')

    assert comment


def test_weekly_update():
    current_datetime = pd.datetime.utcnow()
    start_datetime = current_datetime - pd.Timedelta(days=10)
    end_datetime = start_datetime + pd.Timedelta(days=4)

    steem_per_mvests = 490
    total_posts = 70000
    total_votes = 99897788
    total_reward = 79898973

    bid_bots_sbd = 4242
    bid_bots_steem = 12
    bid_bots_percent = 99.9998

    median_reward = 0.012
    mean_reward = 6.2987347329
    dollar_percent = 69.80921393
    spelling_percent = 1.435
    style_percent = 5.5
    topic_percent = 100 - style_percent - spelling_percent

    top_posts_authors = ['michael', 'mary', 'lala']
    top_posts_titles = ['How', 'What', 'Why']
    top_posts_rewards = [9.999, 6.6, 3.333]
    top_posts_permalinks = ['how', 'what', 'why']

    top_tags = ['ketchup', 'crypto']
    top_tag_counts = [10009, 4445]
    top_tag_rewards = [3213323, 413213213]

    top_words = ['a', 'the']
    top_words_counts = [6666, 2222]

    top_tags_earnings=['hi']
    top_tags_earnings_counts=[10]
    top_tags_earnings_reward=[22]
    top_tfidf=['hi']
    top_tfidf_scores=[0.8]

    delegator_list = ['henry', 'mike', 'julia']

    topics = """Topic 0: global: 0.25, report: 0.18, sales: 0.18, research: 0.15, product: 0.13, industry: 0.13, 20132018: 0.12
Topic 1: global: -0.26, sales: -0.22, report: -0.16, 20132018: -0.16, revenue: -0.13, product: -0.13, market share: -0.12
Topic 2: blockchain: -0.23, game: 0.19, data: -0.17, currency: -0.15, sales: 0.13, digital: -0.13, the price: -0.12
Topic 3: game: -0.75, the game: -0.26, this game: -0.18, play: -0.16, games: -0.13, game is: -0.08, to play: -0.08
Topic 4: report: -0.22, volume: 0.20, sales: 0.20, research: -0.15, global: -0.15, game: -0.14, the price: 0.14
Topic 5: report: -0.28, blockchain: 0.28, sales: 0.20, research: -0.14, global: -0.12, network: 0.11, the report: -0.11
Topic 6: fruit: 0.27, water: 0.21, dragon: 0.18, dragon fruit: 0.17, steem: -0.15, food: 0.14, health: 0.13
Topic 7: states: -0.30, united: -0.29, united states: -0.28, 20132018: 0.22, global: 0.21, 20122017: -0.16, product: -0.15
Topic 8: steem: -0.60, content: -0.18, posts: -0.16, sbd: -0.11, god: 0.10, government: 0.09, steem power: -0.09
Topic 9: blockchain: -0.44, data: -0.24, god: -0.17, crypto: 0.16, trading: 0.15, network: -0.14, exchanges: 0.13
Topic 10: fruit: -0.42, dragon: -0.33, dragon fruit: -0.31, blockchain: -0.16, water: 0.12, energy: 0.11, fruit is: -0.10
Topic 11: production: 0.18, 20132025: 0.17, analysis: -0.16, the market: -0.15, states: 0.14, status: 0.14, status and: 0.14
Topic 12: god: -0.47, steem: -0.25, water: 0.18, man: -0.15, nature: -0.15, of god: -0.13, blockchain: 0.10
Topic 13: steem: 0.40, water: 0.36, blockchain: 0.20, fruit: -0.12, heat: 0.12, dragon fruit: -0.10, dragon: -0.10
Topic 14: god: 0.37, government: -0.19, water: 0.16, steem: -0.13, coin: 0.11, coins: 0.11, mining: 0.11
Topic 15: cancer: -0.31, fruit: 0.23, dragon: 0.20, dragon fruit: 0.19, energy: 0.16, breast: -0.16, risk: -0.14
Topic 16: equipment: -0.39, sterilization: -0.37, heat: -0.37, moist: -0.36, devices: 0.29, station: 0.27, base: 0.27
Topic 17: blockchain: 0.30, data: -0.25, mining: -0.21, network: -0.18, token: 0.18, transaction: -0.17, users: -0.16
Topic 18: mining: -0.27, energy: -0.22, devices: 0.16, users: 0.15, base: 0.15, lte: 0.15, station: 0.15
Topic 19: devices: -0.32, station: -0.30, lte: -0.29, base: -0.29, equipment: -0.22, sterilization: -0.20, heat: -0.20"""

    title, body = tbpo.weekly_update(current_datetime=current_datetime,
                  steem_per_mvests=steem_per_mvests,
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
                  top_posts_authors=top_posts_authors,
                  top_posts_titles=top_posts_titles,
                  top_posts_rewards=top_posts_rewards,
                  top_posts_permalinks=top_posts_permalinks,
                  top_tags=top_tags,
                  top_tag_counts=top_tag_counts,
                  top_tag_rewards=top_tag_rewards,
                  top_tags_earnings=top_tags_earnings,
                  top_tags_earnings_counts=top_tags_earnings_counts,
                  top_tags_earnings_reward=top_tags_earnings_reward,
                  top_words=top_words,
                  top_words_counts=top_words_counts,
                  top_tfidf=top_tfidf,
                  top_tfidf_scores=top_tfidf_scores,
                  spelling_percent=spelling_percent,
                  style_percent=style_percent,
                  topic_percent=topic_percent,
                  delegator_list=delegator_list,
                  topics=topics)

    assert body
    assert title


def test_top_trending_post():
    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()
    df.image_urls = df.body.apply(lambda x: tptf.get_image_urls(x))

    title, post = tbpo.top_trending_post(topN_authors=df.author,
                                 topN_permalinks=df.permalink,
                                 topN_titles=df.title,
                                 topN_filtered_bodies=df.filtered_body,
                                 topN_image_urls=df.image_urls,
                                 topN_rewards=df.reward, sbd_amount=10,
                                 steem_amount=10,
                                 title_date=date, trufflepicks_link='de.de.de',
                                 truffle_link='www.de')

    assert post
    assert title