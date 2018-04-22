import pandas as pd

from trufflepig.testutils.pytest_fixtures import steem, noapisteem
import trufflepig.bchain.getaccountdata as tpac


def test_find_index_offset(noapisteem):
    now = pd.datetime.utcnow()
    target = now - pd.Timedelta(days=42)
    offset, datetime = tpac.find_nearest_index(target, 'cheetah', noapisteem)
    assert 0 < offset
    assert abs((target - datetime).seconds) < 3600*48


def test_shares_query(noapisteem):
    result = tpac.get_delegates_and_shares('trufflepig', noapisteem)

    assert 'smcaterpillar' in result


def test_payouts(noapisteem):
    now_24 = pd.datetime.utcnow() + pd.Timedelta(days=1)
    result = tpac.get_delegate_payouts('trufflepig', noapisteem,
                                       now_24,
                                       1,
                                       0.5)

    assert 'smcaterpillar' in result
    assert 'trufflepig' not in result


def test_bidbot_test(noapisteem):
    min_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    max_datetime = min_datetime + pd.Timedelta(days=13)
    result = tpac.get_upvote_payments('brittuf', noapisteem, min_datetime,
                                      max_datetime)
    assert result


def test_bidbot_test_max_time(noapisteem):
    min_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    max_datetime = min_datetime + pd.Timedelta(days=13)
    result = tpac.get_upvote_payments('brittuf', noapisteem, min_datetime,
                                      max_datetime, max_time=0.1)
    assert len(result) <= 1


def test_get_upvote_payments_for_accounts(noapisteem):
    min_datetime = pd.datetime.utcnow() - pd.Timedelta(days=14)
    max_datetime = min_datetime + pd.Timedelta(days=5)
    accounts = ['trufflepig', 'smcaterpillar', 'brittuf']
    result = tpac.get_upvote_payments_for_accounts(accounts,
                                                   noapisteem,
                                                   min_datetime=min_datetime,
                                                   max_datetime=max_datetime)
    assert result
