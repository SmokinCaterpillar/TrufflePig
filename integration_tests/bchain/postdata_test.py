import pandas as pd
import pytest

import trufflepig.bchain.getdata as tpbg
import trufflepig.bchain.postdata as tbpd
import trufflepig.preprocessing as tppp
from trufflepig import config
from trufflepig.testutils import random_data
from trufflepig.testutils.pytest_fixtures import steem


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_top10post(steem):

    steem.wallet.unlock(config.PASSWORD)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df['predicted_reward'] = df.reward
    df['predicted_votes'] = df.votes

    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()

    account = config.ACCOUNT

    permalink = tbpd.post_topN_list(df, steem, account, date,
                                    overview_permalink='iii')
    tbpd.comment_on_own_top_list(df, steem, account, permalink)


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_all_top_with_real_data(steem):

    steem.wallet.unlock(config.PASSWORD)

    df = tpbg.scrape_hour_data(steem, stop_after=10)

    df = tppp.preprocess(df)

    df['predicted_reward'] = df.reward
    df['predicted_votes'] = df.votes

    date = pd.datetime.utcnow().date()

    account = config.ACCOUNT

    permalink = tbpd.post_topN_list(df, steem, account, date,
                                    overview_permalink='jjj')
    tbpd.comment_on_own_top_list(df, steem, account, permalink)
    tbpd.vote_and_comment_on_topK(df, steem, account, permalink, K=1,
                                  overview_permalink='jjj')


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_top20_vote_and_comment(steem):

    steem.wallet.unlock(config.PASSWORD)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df['predicted_reward'] = df.reward
    df['predicted_votes'] = df.votes

    df = tppp.preprocess(df, ncores=1)

    account = config.ACCOUNT

    tbpd.vote_and_comment_on_topK(df, steem, account, 'laida',
                                  overview_permalink='lll')


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_create_wallet(steem):
    tbpd.create_wallet(steem, config.PASSWORD, config.POSTING_KEY)