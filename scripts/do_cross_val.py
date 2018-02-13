import logging
import os

from steem import Steem

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
from trufflepig import config


def main():
    logging.basicConfig(level=logging.INFO)
    directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')

    steem = dict(nodes=[config.NODE_URL])
    current_datetime = '2018-02-01'
    post_frame = tpgd.scrape_or_load_training_data(steem, directory,
                                                   current_datetime=current_datetime,
                                                   days=3,
                                                   offset_days=0)

    regressor_kwargs = dict(n_estimators=20, max_leaf_nodes=100,
                              max_features=0.1, n_jobs=-1, verbose=1,
                              random_state=42)

    topic_kwargs = dict(num_topics=50, no_below=5, no_above=0.7)

    post_frame = tppp.preprocess(post_frame, ncores=4, chunksize=20)

    param_grid = {
        'feature_generation__topic_model__no_above':[0.2, 0.3],
        'regressor__max_leaf_nodes': [50, 100],
        }

    tpmo.cross_validate(post_frame, param_grid, topic_kwargs=topic_kwargs,
                        regressor_kwargs=regressor_kwargs)

if __name__ == '__main__':
    main()