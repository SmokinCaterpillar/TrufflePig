import pandas as pd

from integration_tests.bchain.getdata_test import steem, steem_kwargs

import trufflepig.bchain.getaccountdata as tpac


def test_find_index_offset(steem):
    now = pd.datetime.utcnow()
    target = now - pd.Timedelta(days=42)
    offset, datetime = tpac.find_nearest_index(target, 'cheetah', steem)
    assert 0 < offset
    assert abs((target - datetime).seconds) < 3600*48
