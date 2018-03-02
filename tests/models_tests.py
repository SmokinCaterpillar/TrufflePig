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


def test_create_ngrams():
    expected = ['hello world', 'world peace', 'peace corps']
    result = list(tpmo.create_ngrams(['hello', 'world', 'peace', 'corps'], n=2))
    assert expected == result

    expected = ['hello world peace corps']
    result = list(tpmo.create_ngrams(['hello', 'world', 'peace', 'corps'], n=4))
    assert expected == result

    expected = []
    result = list(tpmo.create_ngrams(['hello', 'world', 'peace', 'corps'], n=5))
    assert expected == result

    expected = ['hello', 'world', 'peace', 'corps']
    result = tpmo.create_ngrams(['hello', 'world', 'peace', 'corps'], n=1)
    assert expected == result
