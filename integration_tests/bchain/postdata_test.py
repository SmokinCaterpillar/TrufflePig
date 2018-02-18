import pytest
import pandas as pd

from tests.fixtures import random_data
from integration_tests.bchain.getdata_test import steem, steem_kwargs

import trufflepig.bchain.posts as tbpo
import trufflepig.bchain.postdata as tbpd
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpbg
from trufflepig import config


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_top10post(steem):

    steem.wallet.unlock(config.PASSWORD)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()

    account = config.ACCOUNT

    tbpd.post_topN_list(df, steem, account, date)


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_top20_vote_and_comment(steem):

    steem.wallet.unlock(config.PASSWORD)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    account = config.ACCOUNT

    tbpd.vote_and_comment_on_topK(df, steem, account, 'laida')


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_create_wallet(steem):
    tbpd.create_wallet(steem, config.PASSWORD, config.POSTING_KEY)