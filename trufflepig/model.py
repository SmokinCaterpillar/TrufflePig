import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline

from gensim.models.lsimodel import LsiModel
from gensim import corpora
from gensim.matutils import corpus2dense


logger = logging.getLogger(__name__)


FEATURES = ['body_length', 'num_paragraphs', 'num_paragraphs',
            'num_spelling_errors', 'chars_per_word', 'words_per_paragraph',
            'errors_per_word', 'average_sentence_length', 'sentence_length_variance']

TARGETS = ['reward', 'votes']


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


def train_pipeline(post_frame, topic_kwargs, regressor_kwargs,
                 features=FEATURES, targets=TARGETS):

    logger.info('Training pipeline...')
    target_frame = post_frame.loc[:, targets]

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

    pipeline.fit(post_frame, target_frame)

    score = pipeline.score(post_frame, target_frame)
    logger.info('...Done! Training score {}'.format(score))

    return pipeline


######## Deprecated ############

# def train_models(post_frame, topic_kwargs, regressor_kwargs,
#                  features=FEATURES, targets=TARGETS):
#
#     logger.info('Training topic model with {}'.format(topic_kwargs))
#     topic_model = TopicModel(**topic_kwargs)
#     topic_model.train(post_frame.tokens)
#
#     X_topic = topic_model.project_dense(post_frame.tokens)
#     X_style = post_frame.loc[:, features].values
#     X = np.concatenate((X_topic, X_style), axis=1)
#
#     logger.info('Training data shape {}'.format(X.shape))
#
#     Y = post_frame.loc[:, targets].values
#
#     logger.info('Target data shape {}.\nTraining regressor'.format(Y.shape))
#
#     regressor = RandomForestRegressor(**regressor_kwargs)
#     regressor.fit(X, Y)
#
#     score = regressor.score(X, Y)
#     logger.info('Training score {}'.format(score))
#
#     return topic_model, regressor