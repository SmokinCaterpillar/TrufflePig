import pandas as pd

import trufflepig.preprocessing as tppp
from trufflepig.testutils.random_data import create_n_random_posts
from trufflepig.testutils.raw_data import POSTS


def test_preprocessing():
    post_frame = pd.DataFrame(POSTS)
    filtered = tppp.preprocess(post_frame, ncores=1, min_en_prob=0.5,
                               max_errors_per_word=0.5,
                               min_max_num_words=(10, 99999))

    assert len(filtered)


def test_preprocessing_parallel():
    post_frame = pd.DataFrame([POSTS[0] for _ in range(100)])
    post_frame['permalink'] = ['kkk'+str(irun % 50) for irun in range(100)]
    filtered = tppp.preprocess(post_frame, ncores=5, chunksize=20,
                               min_en_prob=0.5, max_errors_per_word=0.5,
                               min_max_num_words=(10, 99999))

    assert len(filtered) > 40


def test_preprocessing_random_parallel():
    posts = create_n_random_posts(50)
    post_frame = pd.DataFrame(posts)
    filtered = tppp.preprocess(post_frame, ncores=5, chunksize=10,
                               min_en_prob=0.5, max_errors_per_word=0.5,
                               min_max_num_words=(10, 99999))

    assert len(filtered) > 20


def test_bid_bot_correction():
    posts = create_n_random_posts(30)
    post_frame = pd.DataFrame(posts)

    bought = {}
    bought[('hello', 'kitty')] = ['19 STEEM']
    sample_frame = post_frame[['author', 'permalink']].sample(10)
    for _, (author, permalink) in sample_frame.iterrows():
        bought[(author, permalink)] = {'aaa':{'amount': '3 STEEM'},
                                       'bbb': {'amount': '4 SBD'}}

    post_frame = tppp.compute_bidbot_correction(post_frame,
                                                bought)

    assert post_frame.adjusted_reward.mean() < post_frame.reward.mean()
    assert all(post_frame.adjusted_reward >= 0)
    assert post_frame.adjusted_votes.mean() < post_frame.votes.mean()
    assert all(post_frame.adjusted_votes >= 0)
