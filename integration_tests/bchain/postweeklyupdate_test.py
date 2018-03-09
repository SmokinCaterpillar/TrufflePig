import pandas as pd
import pytest

import trufflepig.bchain.posts as tpbp
import trufflepig.bchain.postweeklyupdate as tppw
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
from trufflepig import config
from trufflepig.testutils.random_data import create_n_random_posts
from trufflepig.testutils.pytest_fixtures import steem_kwargs


def test_statistics():
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)
    current_date = pd.datetime.utcnow()

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    pipeline = tpmo.train_pipeline(post_frame, topic_kwargs=topic_kwargs,
                                    regressor_kwargs=regressor_kwargs)
    stats = tppw.compute_weekly_statistics(post_frame, pipeline)
    steem_per_mvests = 490

    delegator_list = ['peter', 'paul']

    title, body = tpbp.weekly_update(steem_per_mvests=steem_per_mvests,
                                     current_datetime=current_date,
                                     delegator_list=delegator_list,
                                     **stats)
    assert title
    assert body


def test_existence(steem_kwargs):
    result = tppw.return_overview_permalink_if_exists(account=config.ACCOUNT,
                                                      steem_args=steem_kwargs,
                                                      current_datetime=pd.datetime.utcnow())
    assert isinstance(result, str)


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_weekly_post(steem_kwargs):
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)
    current_date = pd.datetime.utcnow()

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    pipeline = tpmo.train_pipeline(post_frame, topic_kwargs=topic_kwargs,
                                    regressor_kwargs=regressor_kwargs)

    permalink = tppw.post_weakly_update(pipeline, post_frame,
                                        account=config.ACCOUNT,
                                        steem_args=steem_kwargs,
                                        current_datetime=current_date)

    assert permalink