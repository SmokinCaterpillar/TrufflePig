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


def test_get_parent_posts(steem):
    comments_ap = (('lextenebris' ,'re-smcaterpillar-re-lextenebris-re-smcaterpillar-re-lextenebris-re-boucaron-re-lextenebris-re-boucaron-re-lextenebris-programming-digging-in-the-db-and-hitting-a-wall-20180301t230012700z'),
                  ('smcaterpillar', 're-trufflepig-re-trufflepig-a-bot-based-on-natural-language-processing-and-machine-learning-to-support-content-curators-and-minnows-20180227t172111-20180227t172607601z'))

    posts = tpcd.get_parent_posts(comments_ap, steem)

    assert len(posts) == 2
