import os

from trufflepig import config
import trufflepig.bchain.getdata as tpbg


directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')


frames = tpbg.scrape_or_load_training_data_parallel([config.NODE_URL],
                                                       directory,
                                                       days=20,
                                                       stop_after=100,
                                                       ncores=5,
                                                       current_datetime='2018-02-11')