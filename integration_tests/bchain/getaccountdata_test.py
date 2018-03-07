import pandas as pd

from integration_tests.bchain.getdata_test import steem, steem_kwargs

import trufflepig.bchain.getaccountdata as tpac


def test_find_index_offset(steem):
    now = pd.datetime.utcnow()
    target = now - pd.Timedelta(days=42)
    offset, datetime = tpac.find_nearest_index(target, 'cheetah', steem)
    assert 0 < offset
    assert abs((target - datetime).seconds) < 3600*48


def test_shares_query(steem):
    result = tpac.get_delegates_and_shares('trufflepig', steem)

    assert 'smcaterpillar' in result


def test_payouts(steem):
    now_24 = pd.datetime.utcnow() + pd.Timedelta(days=1)
    result = tpac.get_delegate_payouts('trufflepig', steem,
                                       now_24,
                                       1,
                                       0.5)

    assert 'smcaterpillar' in result
    assert 'trufflepig' not in result
