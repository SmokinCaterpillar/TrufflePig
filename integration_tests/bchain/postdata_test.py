import pandas as pd
import pytest

import trufflepig.bchain.getdata as tpbg
import trufflepig.bchain.postdata as tbpd
import trufflepig.preprocessing as tppp
from trufflepig import config
from trufflepig.testutils import random_data
from trufflepig.testutils.pytest_fixtures import steem
from trufflepig.bchain.poster import Poster


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_top10post(steem):

    steem.wallet.unlock(config.PASSWORD)

    poster = Poster(steem=steem,
                    account=config.ACCOUNT,
                    waiting_time=0.1)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df['predicted_reward'] = df.reward
    df['predicted_votes'] = df.votes

    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()

    account = config.ACCOUNT

    permalink = tbpd.post_topN_list(df, poster, date,
                                    overview_permalink='iii')
    tbpd.comment_on_own_top_list(df, poster, permalink)


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_test_all_top_with_real_data(steem):

    steem.wallet.unlock(config.PASSWORD)

    poster = Poster(steem=steem,
                    account=config.ACCOUNT,
                    waiting_time=0.1)

    df = tpbg.scrape_hour_data(steem, stop_after=10)

    df = tppp.preprocess(df)

    df['predicted_reward'] = df.reward
    df['predicted_votes'] = df.votes

    date = pd.datetime.utcnow().date()

    account = config.ACCOUNT

    permalink = tbpd.post_topN_list(df, poster, date,
                                    overview_permalink='jjj')
    tbpd.comment_on_own_top_list(df, poster, permalink)
    tbpd.vote_and_comment_on_topK(df, poster, permalink, K=1,
                                  overview_permalink='jjj')


def test_test_top20_vote_and_comment(steem):

    steem.wallet.unlock(config.PASSWORD)

    poster = Poster(steem=steem,
                    account=config.ACCOUNT,
                    waiting_time=0.1,
                    no_posting_key_mode=config.PASSWORD is None)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df['predicted_reward'] = df.reward
    df['predicted_votes'] = df.votes

    df = tppp.preprocess(df, ncores=1)

    tbpd.vote_and_comment_on_topK(df, poster, 'laida',
                                  overview_permalink='lll')


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_create_wallet(steem):
    tbpd.create_wallet(steem, config.PASSWORD, config.POSTING_KEY)


def test_test_top_trending_post(steem):

    steem.wallet.unlock(config.PASSWORD)

    poster = Poster(steem=steem,
                    account=config.ACCOUNT,
                    waiting_time=0.1,
                    no_posting_key_mode=config.PASSWORD is None)

    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df['reward'] = df.reward
    df['predicted_votes'] = df.votes

    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()

    tbpd.post_top_trending_list(df, poster, date,
                                overview_permalink='iii',
                                trufflepicks_permalink='kkk',
                                steem_amount=10,
                                sbd_amount=10)
