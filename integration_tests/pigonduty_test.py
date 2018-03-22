import pytest

from trufflepig.testutils.pytest_fixtures import steem
from integration_tests.model_test import MockPipeline

from trufflepig import config
import trufflepig.pigonduty as tppd


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_call_a_pig(steem):
    current_datetime = '2018-03-03-18:21:30'

    pipeline = MockPipeline()
    tppd.call_a_pig(steem=steem,account='trufflepig',
                    pipeline=pipeline, topN_permalink='www.test.com',
                    current_datetime=current_datetime, hours=0.1,
                    sleep_time=0.1, overview_permalink='dsfd')


@pytest.mark.skipif(config.PASSWORD is None, reason="needs posting key")
def test_call_a_pig_empty_frame(steem):
    aacs = (('smcaterpillar','question-is-there-an-api-to-upload-images-to-steemit'),)

    pipeline = MockPipeline()
    tppd.execute_call(comment_authors_and_permalinks=aacs,
                        steem=steem,account='trufflepig',
                        pipeline=pipeline, topN_permalink='www.test.com',
                        max_comments=1000, sleep_time=0.1,
                        overview_permalink='jdsakd')
