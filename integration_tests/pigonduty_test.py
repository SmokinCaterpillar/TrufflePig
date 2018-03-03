from integration_tests.bchain.getdata_test import steem_kwargs
from integration_tests.model_test import MockPipeline


import trufflepig.pigonduty as tppd


def test_call_a_pig(steem_kwargs):
    current_datetime = '2018-03-03-18:21:30'

    pipeline = MockPipeline()
    tppd.call_a_pig(steem_kwargs=steem_kwargs,account='trufflepig',
                    pipeline=pipeline, topN_link='www.test.com',
                    current_datetime=current_datetime, hours=0.1,
                    sleep_time=0.1)
