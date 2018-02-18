import logging
import os
import gc

import pandas as pd

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
import trufflepig.bchain.postdata as tppd
from trufflepig import config


logger = logging.getLogger(__name__)


def main():

    current_datetime = pd.datetime.utcnow()

    format=('%(asctime)s %(processName)s:%(name)s:'
                  '%(funcName)s:%(lineno)s:%(levelname)s: %(message)s')
    logging.basicConfig(level=logging.INFO, format=format)

    logger.info('STARTING main script at {}'.format(current_datetime))

    data_directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')
    model_directoy = os.path.join(config.PROJECT_DIRECTORY, 'trained_models')

    steem_kwargs = dict(nodes=[config.NODE_URL], no_broadcast=True)

    if not tpmo.model_exists(current_datetime, model_directoy):
        post_frame = tpgd.load_or_scrape_training_data(steem_kwargs, data_directory,
                                                       current_datetime=current_datetime,
                                                       days=9,
                                                       offset_days=0)

        post_frame = tppp.preprocess(post_frame)
        logger.info('Garbage collecting')
        gc.collect()
    else:
        post_frame = None


    regressor_kwargs = dict(n_estimators=256, max_leaf_nodes=4096,
                              max_features=0.2, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=128, no_below=5, no_above=0.1)

    pipeline = tpmo.load_or_train_pipeline(post_frame, model_directoy,
                                           current_datetime,
                                           regressor_kwargs=regressor_kwargs,
                                           topic_kwargs=topic_kwargs)

    tpmo.log_pipeline_info(pipeline=pipeline)


    prediction_frame = tpgd.scrape_hour_data(steem_or_args=steem_kwargs,
                                             current_datetime=current_datetime)
    prediction_frame = tppp.preprocess(prediction_frame)

    sorted_frame = tpmo.find_truffles(prediction_frame, pipeline)
    account = config.ACCOUNT
    permalink = tppd.post_topN_list(sorted_frame, steem_kwargs,
                                    account=account,
                                    current_datetime=current_datetime)
    tppd.vote_and_comment_on_topK(sorted_frame,
                                  steem_kwargs,
                                  topN_permalink=permalink,
                                  account=account)

    logger.info('DONE at {}'.format(current_datetime))



if __name__ == '__main__':
    main()
