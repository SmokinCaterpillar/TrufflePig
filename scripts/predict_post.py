import argparse
import os
import logging

import pandas as pd

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
from trufflepig import config


PERMA = 'trufflepig-a-bot-based-on-natural-language-processing-and-machine-learning-to-support-content-curators-and-minnows'
AUTHOR = 'smcaterpillar'


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='TrufflePig Bot')
    parser.add_argument('--author', action='store', default=AUTHOR)
    parser.add_argument('--permalink', action='store', default=PERMA)
    parser.add_argument('--now', action='store', default=None)

    args = parser.parse_args()

    return args.author, args.permalink, args.now



def main():

    logging.basicConfig(level=logging.INFO)

    author, permalink, current_datetime = parse_args()

    if current_datetime is None:
        current_datetime = pd.datetime.utcnow()
    else:
        current_datetime = pd.to_datetime(current_datetime)

    model_directoy = os.path.join(config.PROJECT_DIRECTORY, 'trained_models')

    pipeline = tpmo.load_or_train_pipeline(None, model_directoy,
                                           current_datetime)

    steem_kwargs = dict(nodes=config.NODES, no_broadcast=True)
    steem = tpgd.check_and_convert_steem(steem_kwargs)
    posts = tpgd.get_post_data([(author, permalink)], steem, {})

    posts = pd.DataFrame(posts)

    post_frame = tppp.preprocess(posts)

    tpmo.find_truffles(post_frame, pipeline, max_grammar_errors_per_sentence=0.9)



if __name__ == '__main__':
    main()
