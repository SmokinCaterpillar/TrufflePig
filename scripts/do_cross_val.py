import logging
import os
import gc

from steem import Steem

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
from trufflepig import config


def main():

    format=('%(asctime)s %(processName)s:%(name)s:'
                  '%(funcName)s:%(lineno)s:%(levelname)s: %(message)s')
    logging.basicConfig(level=logging.INFO, format=format)
    directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')

    steem = dict(nodex=config.NODES)
    current_datetime = '2018-02-01'

    crossval_filename = os.path.join(directory, 'xval_{}.gz'.format(current_datetime))

    post_frame = tpgd.load_or_scrape_training_data(steem, directory,
                                                   current_datetime=current_datetime,
                                                   days=9,
                                                   offset_days=0)

    gc.collect()

    regressor_kwargs = dict(n_estimators=256, max_leaf_nodes=4096,
                              max_features=0.2, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=128, no_below=5, no_above=0.1)

    post_frame = tppp.load_or_preprocess(post_frame, crossval_filename,
                                         ncores=4, chunksize=1000,
                                         min_en_prob=0.9)
    gc.collect()
    param_grid = {
        'feature_generation__topic_model__no_above':[0.33],
        #'regressor__max_leaf_nodes': [500, 1000],
       # 'regressor__max_features': [0.1, 0.2, 0.3]
        }

    # tpmo.cross_validate(post_frame, param_grid, topic_kwargs=topic_kwargs,
    #                     regressor_kwargs=regressor_kwargs, n_iter=None,
    #                     n_jobs=4, targets=['reward'])

    pipe, test_frame = tpmo.train_test_pipeline(post_frame,  topic_kwargs=topic_kwargs,
                         regressor_kwargs=regressor_kwargs, targets=['reward', 'votes'])

    topic_model = pipe.named_steps['feature_generation'].transformer_list[1][1]
    logging.getLogger().info(topic_model.print_topics(n_best=None))

    tpmo.find_truffles(test_frame, pipe, min_votes=5)


if __name__ == '__main__':
    main()