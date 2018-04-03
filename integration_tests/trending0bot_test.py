import pytest

import pandas as pd
import trufflepig.bchain.getdata as tpgd
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getaccountdata as tpad
from trufflepig.bchain.poster import Poster
import trufflepig.trending0bidbots as tt0b
from trufflepig.testutils.pytest_fixtures import steem
from trufflepig import config


def test_create_trending_post(steem):

    current_datetime = pd.datetime.utcnow()

    data_frame = tpgd.scrape_hour_data(steem=steem,
                                             current_datetime=current_datetime,
                                             ncores=32,
                                             offset_hours=8,
                                       hours=1, stop_after=20)


    min_datetime = data_frame.created.min()
    max_datetime = data_frame.created.max() + pd.Timedelta(days=8)
    upvote_payments = tpad.get_upvote_payments_to_bots(steem=steem,
                                                  min_datetime=min_datetime,
                                                  max_datetime=max_datetime,
                                                 bots=['booster'])

    data_frame = df = tppp.preprocess(data_frame, ncores=1)

    data_frame = tppp.compute_bidbot_correction(post_frame=data_frame,
                                                upvote_payments=upvote_payments)
    account = config.ACCOUNT
    poster = Poster(account=account, steem=steem,
                    no_posting_key_mode=config.PASSWORD is None)

    tt0b.create_trending_post(data_frame, upvote_payments, poster, 'test',
                         'test', current_datetime)


def test_bottracker_api(steem):

    min_datetime = pd.datetime.utcnow() - pd.Timedelta(minutes=30)
    max_datetime = pd.datetime.utcnow()
    upvote_payments = tpad.get_upvote_payments_to_bots(steem=steem,
                                                  min_datetime=min_datetime,
                                                  max_datetime=max_datetime,
                                                 bots='default')
    assert upvote_payments
