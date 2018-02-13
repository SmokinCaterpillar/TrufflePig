import os

import pandas as pd
import pytest

from tests.fixtures.random_data import create_n_random_posts
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp


@pytest.fixture
def temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test', numbered=True)


def test_pipeline_model():
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    tpmo.train_pipeline(post_frame, **dict(topic_kwargs=topic_kwargs,
                                            regressor_kwargs=regressor_kwargs))


def test_load_or_train(temp_dir):
    cdt = pd.datetime.utcnow()
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)

    pipe = tpmo.load_or_train_pipeline(post_frame, temp_dir,
                                       current_datetime=cdt,
                                       **dict(topic_kwargs=topic_kwargs,
                                              regressor_kwargs=regressor_kwargs))

    assert len(os.listdir(temp_dir)) == 1

    pipe2 = tpmo.load_or_train_pipeline(post_frame, temp_dir,
                                        current_datetime=cdt,
                                       **dict(topic_kwargs=topic_kwargs,
                                              regressor_kwargs=regressor_kwargs))

    assert len(os.listdir(temp_dir)) == 1
    assert set(pipe.named_steps.keys()) == set(pipe2.named_steps.keys())

