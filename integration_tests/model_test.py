import pandas as pd

from tests.fixtures.random_data import create_n_random_posts
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp


def test_train_model():
    posts = create_n_random_posts(300)

    post_frame = pd.DataFrame(posts)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwards = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=50)
    t, r = tpmo.train_models(post_frame, topic_kwards, regressor_kwargs)