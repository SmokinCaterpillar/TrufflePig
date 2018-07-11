from trufflepig.testutils.pytest_fixtures import steem
import trufflepig.pigonduty as tppd
from tests.integration.model_test import MockPipeline
from trufflepig import config
from trufflepig.bchain.poster import Poster


def test_call_a_pig(steem):
    current_datetime = '2018-03-03-18:21:30'

    pipeline = MockPipeline()
    poster = Poster(steem=steem,
                    account=config.ACCOUNT,
                    waiting_time=0.1,
                    no_posting_key_mode=config.PASSWORD is None)

    tppd.call_a_pig(poster=poster,
                    pipeline=pipeline, topN_permalink='www.test.com',
                    current_datetime=current_datetime, hours=0.1,
                    overview_permalink='dsfd')


def test_call_a_pig_empty_frame(steem):
    aacs = (('smcaterpillar','question-is-there-an-api-to-upload-images-to-steemit'),)

    poster = Poster(steem=steem,
                    account=config.ACCOUNT,
                    waiting_time=0.51,
                    no_posting_key_mode=config.PASSWORD is None)

    pipeline = MockPipeline()
    tppd.execute_call(comment_authors_and_permalinks=aacs,
                        pipeline=pipeline, topN_permalink='www.test.com',
                        max_comments=1000,
                        overview_permalink='jdsakd',
                      poster=poster)
