import pytest
import pandas as pd

from trufflepig.testutils.pytest_fixtures import steem, noapisteem
import trufflepig.bchain.paydelegates as tppd
import trufflepig.bchain.postdata as tpdd
from trufflepig import config


@pytest.mark.skipif(config.ACTIVE_KEY is None, reason="needs active key")
def test_pay_delegates(noapisteem):

    tpdd.create_wallet(noapisteem, config.PASSWORD,
                       config.POSTING_KEY, config.ACTIVE_KEY)

    tppd.pay_delegates(account=config.ACCOUNT,
                       steem=noapisteem,
                       current_datetime=pd.datetime.utcnow()#'2029-01-01'
                       )
