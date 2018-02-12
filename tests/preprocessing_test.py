import pandas as pd

from tests.fixtures.raw_data import POSTS

import trufflepig.preprocessing as tppp


def test_preprocessing():
    post_frame = pd.DataFrame(POSTS)
    filtered = tppp.preprocess(post_frame, ncores=1)


def test_preprocessing_parallel():
    post_frame = pd.DataFrame([POSTS[0] for _ in range(1000)])
    post_frame['permalink'] = ['kkk'+str(irun % 500) for irun in range(1000)]
    filtered = tppp.preprocess(post_frame, ncores=7)

    assert len(filtered) > 400