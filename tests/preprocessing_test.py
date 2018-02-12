import pandas as pd

from tests.fixtures.raw_data import POSTS

import trufflepig.preprocessing as tppp


def test_preprocessing():
    post_frame = pd.DataFrame(POSTS)
    filtered = tppp.preprocess(post_frame, ncores=1)

    assert len(filtered)


def test_preprocessing_parallel():
    post_frame = pd.DataFrame([POSTS[0] for _ in range(100)])
    post_frame['permalink'] = ['kkk'+str(irun % 50) for irun in range(100)]
    filtered = tppp.preprocess(post_frame, ncores=5, chunksize=20)

    assert len(filtered) > 40