import pandas as pd
import numpy as np

from tests.fixtures.random_data import create_n_random_posts
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp


def test_tag_measure():
    posts = create_n_random_posts(100)

    post_frame = pd.DataFrame(posts)

    post_frame = tppp.preprocess(post_frame, ncores=1)

    post_frame['predicted_reward'] = post_frame.reward

    tag_measure = tpmo.compute_tag_factor(post_frame.tags, tpmo.PUNISH_LIST)

    assert np.all(tag_measure > 0)
