import pandas as pd

from integration_tests.bchain.getdata_test import steem_kwargs, steem

import trufflepig.bchain.checkops as tpcd


def test_check_all_ops_between(steem):
    start = pd.to_datetime('2018-01-17-13:39:00')
    end = pd.to_datetime('2018-01-17-13:41:00')
    comments = tpcd.check_all_ops_between(start, end, steem,
                                       account='originalworks',
                                       stop_after=15)
    assert comments


def test_check_all_ops_between_parallel(steem_kwargs):
    start = pd.to_datetime('2018-01-17-13:39:00')
    end = pd.to_datetime('2018-01-17-13:41:00')
    comments = tpcd.check_all_ops_between_parallel(start, end, steem_kwargs,
                                       account='originalworks',
                                       stop_after=15, ncores=4)
    assert comments
