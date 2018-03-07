import pandas as pd

from tests.fixtures.raw_data import POSTS
from tests.fixtures.random_data import create_n_random_posts

import trufflepig.preprocessing as tppp


def test_preprocessing():
    post_frame = pd.DataFrame(POSTS)
    filtered = tppp.preprocess(post_frame, ncores=1, min_en_prob=0.8,
                               max_errors_per_word=0.5)

    assert len(filtered)


def test_preprocessing_parallel():
    post_frame = pd.DataFrame([POSTS[0] for _ in range(100)])
    post_frame['permalink'] = ['kkk'+str(irun % 50) for irun in range(100)]
    filtered = tppp.preprocess(post_frame, ncores=5, chunksize=20,
                               min_en_prob=0.8, max_errors_per_word=0.5)

    assert len(filtered) > 40


def test_preprocessing_random_parallel():
    posts = create_n_random_posts(50)
    post_frame = pd.DataFrame(posts)
    filtered = tppp.preprocess(post_frame, ncores=5, chunksize=10,
                               min_en_prob=0.8, max_errors_per_word=0.5)

    assert len(filtered) > 30
