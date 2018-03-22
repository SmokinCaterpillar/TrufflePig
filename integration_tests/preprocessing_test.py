import os

import pandas as pd
from pandas.testing import assert_frame_equal

import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
from trufflepig.testutils.random_data import create_n_random_posts
from trufflepig.testutils.pytest_fixtures import temp_dir, steem


def test_load_or_preproc(temp_dir):
    filename = os.path.join(temp_dir, 'pptest.gz')

    post_frame = pd.DataFrame(create_n_random_posts(10))

    frame = tppp.load_or_preprocess(post_frame, filename,
                                    ncores=5, chunksize=20)

    assert len(os.listdir(temp_dir)) == 1

    frame2 = tppp.load_or_preprocess(post_frame, filename,
                                    ncores=5, chunksize=20)

    assert len(os.listdir(temp_dir)) == 1
    assert_frame_equal(frame, frame2)


def test_load_or_preproc_with_real_data(steem, temp_dir):
    filename = os.path.join(temp_dir, 'pptest.gz')

    start_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    end_datetime = start_datetime + pd.Timedelta(hours=2)
    posts = tpgd.get_all_posts_between_parallel(start_datetime,
                                                     end_datetime,
                                                     steem,
                                                     stop_after=15)
    post_frame = pd.DataFrame(posts)
    bots = ['okankarol', 'bidseption', 'highvote', 'oguzhangazi', 'ottoman',]
    frame = tppp.load_or_preprocess(post_frame, filename,
                                    steem_args_for_upvote=steem,
                                    ncores=5, chunksize=20, bots=bots)

    assert len(os.listdir(temp_dir)) == 1

    frame2 = tppp.load_or_preprocess(post_frame, filename,
                                    steem_args_for_upvote=steem,
                                    ncores=5, chunksize=20, bots=bots)

    assert len(os.listdir(temp_dir)) == 1
    assert_frame_equal(frame, frame2)
