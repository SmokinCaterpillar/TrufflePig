import os

from pandas.testing import assert_frame_equal
import pandas as pd

from integration_tests.model_test import temp_dir
from tests.fixtures.random_data import create_n_random_posts
import trufflepig.preprocessing as tppp


def test_load_or_preproc(temp_dir):
    filename = os.path.join(temp_dir, 'pptest.gz')

    post_frame = pd.DataFrame(create_n_random_posts(10))

    frame = tppp.load_or_preprocess(post_frame, filename,
                                    ncores=5, chunksize=20)

    assert len(os.listdir(temp_dir)) == 1

    frame2 = tppp.load_or_preprocess(post_frame, filename,
                                    ncores=5, chunksize=20,
                                    max_grammar_errors_per_sentence=10,
                                    grammar_max_sentences=2)

    assert len(os.listdir(temp_dir)) == 1
    assert_frame_equal(frame, frame2)