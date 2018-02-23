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
    pipe = tpmo.train_pipeline(post_frame, topic_kwargs=topic_kwargs,
                                    regressor_kwargs=regressor_kwargs)

    tpmo.log_pipeline_info(pipe)


def test_train_test_pipeline():
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    tpmo.train_test_pipeline(post_frame, topic_kwargs=topic_kwargs,
                                    regressor_kwargs=regressor_kwargs)


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
                                       topic_kwargs=topic_kwargs,
                                       regressor_kwargs=regressor_kwargs)

    topic_model = pipe.named_steps['feature_generation'].transformer_list[1][1]
    result = topic_model.print_topics()
    assert result

    assert len(os.listdir(temp_dir)) == 1

    pipe2 = tpmo.load_or_train_pipeline(post_frame, temp_dir,
                                        current_datetime=cdt,
                                        topic_kwargs=topic_kwargs,
                                        regressor_kwargs=regressor_kwargs)

    assert len(os.listdir(temp_dir)) == 1
    assert set(pipe.named_steps.keys()) == set(pipe2.named_steps.keys())


def test_Doc2Vec_KNN():
    posts = create_n_random_posts(100)

    post_frame = pd.DataFrame(posts)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=30)

    pipe = tpmo.create_pure_doc2vec_pipeline(dict(epochs=2, size=16))

    pipe, frame = tpmo.train_test_pipeline(post_frame, pipeline=pipe,
                                           sample_weight_function=None)
    pass


def test_crossval():
    posts = create_n_random_posts(100)

    post_frame = pd.DataFrame(posts)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=20)

    param_grid = {
        'feature_generation__topic_model__no_above':[0.2, 0.3],
        'regressor__max_leaf_nodes': [50, 100],
        }

    tpmo.cross_validate(post_frame, param_grid, topic_kwargs=topic_kwargs,
                        regressor_kwargs=regressor_kwargs)


def test_find_truffles():
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    pipeline = tpmo.train_pipeline(post_frame, topic_kwargs=topic_kwargs,
                                    regressor_kwargs=regressor_kwargs)

    posts = create_n_random_posts(50)

    post_frame = pd.DataFrame(posts)
    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    truffles = tpmo.find_truffles(post_frame, pipeline, min_max_reward=(0,100),
                                  max_grammar_errors_per_sentence=5)

    assert truffles.iloc[0].reward_difference == truffles.reward_difference.max()