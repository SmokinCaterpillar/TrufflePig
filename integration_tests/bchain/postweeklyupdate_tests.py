import pandas as pd

from tests.fixtures.random_data import create_n_random_posts
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.postweeklyupdate as tppw
import trufflepig.bchain.posts as tpbp


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

    title, body = tpbp.weekly_update(steem_per_mvests=steem_per_mvests,
                                     current_datetime=current_date,
                                     **stats)
    assert title
    assert body
