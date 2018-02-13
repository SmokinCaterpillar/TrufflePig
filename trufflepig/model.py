import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib

from gensim.models.lsimodel import LsiModel
from gensim import corpora
from gensim.matutils import corpus2dense


logger = logging.getLogger(__name__)


FEATURES = ['body_length', 'num_paragraphs', 'num_paragraphs',
            'num_spelling_errors', 'chars_per_word', 'words_per_paragraph',
            'errors_per_word', 'average_sentence_length', 'sentence_length_variance']

TARGETS = ['reward', 'votes']

FILENAME_TEMPLATE = 'truffle_pipeline__{time}.gz'

class TopicModel(object):
    def __init__(self, no_below, no_above, num_topics):
        self.num_topics = num_topics
        self.no_below = no_below
        self.no_above = no_above
        self.lsi = None
        self.disctionary = None

    def to_corpus(self, tokens):
        return [self.dictionary.doc2bow(text) for text in tokens]

    def train(self, tokens):
        self.dictionary = corpora.Dictionary(tokens)
        self.dictionary.filter_extremes(self.no_below, self.no_above)
        corpus = self.to_corpus(tokens)
        self.lsi = LsiModel(corpus, num_topics=self.num_topics)

    def project(self, tokens):
        corpus = self.to_corpus(tokens)
        return self.lsi[corpus]

    def project_dense(self, tokens):
        projection = self.project(tokens)
        result = corpus2dense(projection, self.num_topics).T
        return result

    def fit(self, data, y=None):
        self.train(data.tokens)
        return self

    def transform(self, data):
        return self.project_dense(data.tokens)


class FeatureSelector(object):
    def __init__(self, features):
        self.features = features

    def fit(self, X=None, y=None):
        return self

    def transform(self, data):
        return data.loc[:, self.features]


def create_pipeline(topic_kwargs, regressor_kwargs, features=FEATURES):
    feature_generation = FeatureUnion(
        transformer_list=[
            ('topic_model', TopicModel(**topic_kwargs)),
            ('feature_selection', FeatureSelector(features))
        ]
    )

    pipeline = Pipeline(steps=[
            ('feature_generation', feature_generation),
            ('regressor', RandomForestRegressor(**regressor_kwargs))
        ]
    )

    return pipeline


def train_pipeline(post_frame, **kwargs):
    targets = kwargs.get('targets', TARGETS)

    logger.info('Training pipeline...')
    target_frame = post_frame.loc[:, targets]

    pipeline = create_pipeline(**kwargs)
    pipeline.fit(post_frame, target_frame)

    score = pipeline.score(post_frame, target_frame)
    logger.info('...Done! Training score {}'.format(score))

    return pipeline


# def cross_validate(post_frame, param_grid=None, **kwargs):
#     targets = kwargs.get('targets', TARGETS)


def load_or_train_pipeline(post_frame, directory, current_datetime=None,
                           overwrite=False, store=True, **kwargs):
    if current_datetime is None:
        current_datetime = pd.datetime.utcnow()
    else:
        current_datetime = pd.to_datetime(current_datetime)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename = FILENAME_TEMPLATE.format(time=current_datetime.strftime('%Y-%U'))

    filename = os.path.join(directory,filename)
    if os.path.isfile(filename) and not overwrite:
        logger.info('Found file {} will load it'.format(filename))
        pipeline = joblib.load(filename)
    else:
        logger.info('File {} not found, will start training'.format(filename))
        pipeline = train_pipeline(post_frame, **kwargs)
        if store:
            logger.info('Storing file {} to disk'.format(filename))
            joblib.dump(pipeline, filename, compress=8)
    return pipeline
