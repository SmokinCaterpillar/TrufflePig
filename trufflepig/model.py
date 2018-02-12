import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

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


def train_models(post_frame, topic_kwargs, regressor_kwargs,
                 features=FEATURES, targets=TARGETS):

    logger.info('Training topic model with {}'.format(topic_kwargs))
    topic_model = TopicModel(**topic_kwargs)
    topic_model.train(post_frame.tokens)

    X_topic = topic_model.project_dense(post_frame.tokens)
    X_style = post_frame.loc[:, features].values
    X = np.concatenate((X_topic, X_style), axis=1)

    logger.info('Training data shape {}'.format(X.shape))

    Y = post_frame.loc[:, targets].values

    logger.info('Target data shape {}.\nTraining regressor'.format(Y.shape))

    regressor = RandomForestRegressor(**regressor_kwargs)
    regressor.fit(X, Y)

    score = regressor.score(X, Y)
    logger.info('Training score {}'.format(score))

    return topic_model, regressor