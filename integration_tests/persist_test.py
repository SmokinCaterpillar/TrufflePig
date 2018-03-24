import os
import pandas as pd

from trufflepig.testutils.pytest_fixtures import temp_dir
from trufflepig.testutils.random_data import create_n_random_posts
import trufflepig.preprocessing as tppp
import trufflepig.persist as tppe


def test_store_load_frame_test(temp_dir):
    filename = os.path.join(temp_dir, 'test.sqlite')

    x = pd.DataFrame(create_n_random_posts(42))
    x = tppp.preprocess(x)

    tppe.to_sqlite(x, filename, 'test')

    y = tppe.from_sqlite(filename, 'test')

    pd.testing.assert_frame_equal(x,y)
