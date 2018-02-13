import logging
import os

import pandas as pd
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


    crossval_filename = os.path.join(directory, 'xval_first_proto.gz')

    post_frame = pd.read_pickle('../scraped_data/first_post_set.gz')

    regressor_kwargs = dict(n_estimators=256, max_leaf_nodes=256,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=64, no_below=5, no_above=0.1)

    post_frame['votes'] = 0
    post_frame = tppp.load_or_preprocess(post_frame, crossval_filename,
                                         ncores=4, chunksize=1000)

    # param_grid = {
    #     #'feature_generation__topic_model__no_above':[0.05, 0.1, 0.2, 0.33],
    #     #'feature_generation__topic_model__num_topics':[50, 100, 200],
    #     'regressor__max_leaf_nodes': [50, 100, 200]
    #    # 'regressor__max_features': [0.1, 0.2, 0.3, 0.66]
    #     }
    #
    # tpmo.cross_validate(post_frame, param_grid, topic_kwargs=topic_kwargs,
    #                     regressor_kwargs=regressor_kwargs, n_iter=None,
    #                     n_jobs=4, targets=['reward'])

    pipe = tpmo.train_test_pipeline(post_frame,  topic_kwargs=topic_kwargs,
                         regressor_kwargs=regressor_kwargs, targets=['reward'])

    topic_model = pipe.named_steps['feature_generation'].transformer_list[0][1]
    logging.getLogger().info(topic_model.print_topics())


if __name__ == '__main__':
    main()