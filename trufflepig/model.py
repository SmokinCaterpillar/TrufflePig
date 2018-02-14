import logging
import os
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, \
    RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator

from gensim.models.lsimodel import LsiModel
from gensim import corpora
from gensim.matutils import corpus2dense


logger = logging.getLogger(__name__)


FEATURES = ['body_length',
            'num_sentences',
            'num_paragraphs',
            'num_words',
            'unique_words',
            'unique_ratio',
            'num_spelling_errors',
            'chars_per_word',
            'words_per_paragraph',
            'errors_per_word',
            'average_sentence_length',
            'sentence_length_variance',
            'average_punctuation',
            'connectors_per_sentence']

TARGETS = ['reward', 'votes']

FILENAME_TEMPLATE = 'truffle_pipeline__{time}.gz'

class TopicModel(BaseEstimator):
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

    def print_topics(self, n_best=10, n_words=7):
        if n_best is None:
            n_best = self.num_topics

        result = ''
        for topic in range(n_best):
            best_words = self.lsi.show_topic(topic, 7)
            inwords = [(self.dictionary[int(x[0])], x[1]) for x in best_words]
            wordstring = ', '.join('{}: {:0.2f}'.format(*x) for x in inwords)
            result += 'Topic {}: {}\n'.format(topic, wordstring)
        return result


class FeatureSelector(BaseEstimator):
    def __init__(self, features):
        self.features = features

    def fit(self, X=None, y=None):
        return self

    def transform(self, data):
        return data.loc[:, self.features]


def create_pipeline(topic_kwargs, regressor_kwargs, features=FEATURES):
    logger.info('Using features {}'.format(features))
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


def compute_weights(target_frame):
    return 1 + np.log(1 + target_frame.votes)


def train_pipeline(post_frame, **kwargs):
    targets = kwargs.pop('targets', TARGETS)

    logger.info('Training pipeline with targets {} and {} '
                'samples...'.format(targets, len(post_frame)))
    target_frame = post_frame.loc[:, targets]

    sample_weight = compute_weights(target_frame)

    pipeline = create_pipeline(**kwargs)
    pipeline.fit(post_frame, target_frame,
                 regressor__sample_weight=sample_weight)

    score = pipeline.score(post_frame, target_frame,
                           sample_weight=sample_weight)
    logger.info('...Done! Training score {}'.format(score))

    return pipeline


def train_test_pipeline(post_frame, train_size=0.8, **kwargs):
    train_frame, test_frame = train_test_split(post_frame,
                                               train_size=train_size)
    targets = kwargs.get('targets', TARGETS)
    pipeline = train_pipeline(train_frame, **kwargs)

    target_frame = test_frame.loc[:, targets]

    logger.info('Using test data...')
    sample_weight = compute_weights(target_frame)
    score = pipeline.score(test_frame, target_frame,
                           sample_weight=sample_weight)
    logger.info('...Done! Test score {}'.format(score))
    return pipeline, test_frame


def cross_validate(post_frame, param_grid,
                   train_size=0.8, n_jobs=3,
                   cv=3, verbose=1,
                   n_iter=None, **kwargs):

    targets = kwargs.pop('targets', TARGETS)
    logger.info('Crossvalidating with targets {}...'.format(targets))

    train_frame, test_frame = train_test_split(post_frame,
                                               train_size=train_size)

    pipeline = create_pipeline(**kwargs)
    if n_iter is None:
        grid_search = GridSearchCV(pipeline, param_grid, n_jobs=n_jobs,
                                   cv=cv, refit=True, verbose=verbose)
    else:
        grid_search = RandomizedSearchCV(pipeline, param_grid, n_jobs=n_jobs,
                                         cv=cv, refit=True, verbose=verbose,
                                         n_iter=n_iter)
    logger.info('Starting Grid Search')
    grid_search.fit(train_frame, train_frame.loc[:, targets])

    logger.info("\nGrid scores on development set:\n")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        logger.info("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))

    score = grid_search.score(test_frame, test_frame.loc[:, targets])
    best_estimator = grid_search.best_estimator_

    topic_model = best_estimator.named_steps['feature_generation'].transformer_list[0][1]
    logger.info('Best topics\n{}\n'.format(topic_model.print_topics()))

    logger.info("FINAL TEST Score {} \n of best estimator:\n\n{}".format(score,
                                        best_estimator.get_params()))


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


def find_truffles(post_frame, pipeline, min_reward=1.0, min_votes=5, k=10):
    logger.info('Filtering scraped data')
    post_frame = post_frame.loc[(post_frame.reward >= min_reward) &
                                (post_frame.votes >= min_votes)]

    logger.info('Predicting truffles')
    predicted_rewards_and_votes = pipeline.predict(post_frame)

    post_frame['predicted_reward'] = predicted_rewards_and_votes[:, 0]
    post_frame['predicted_votes'] = predicted_rewards_and_votes[:, 1]
    post_frame['reward_difference'] = post_frame.predicted_reward - post_frame.reward

    post_frame = post_frame.sort_values('reward_difference', ascending=False)

    for irun in range(k):
        row = post_frame.iloc[irun]
        logger.info('\n\n----------------------------------------------'
                    '--------------------------------------------------'
                    '\n############ {} ############'.format(row.title))
        logger.info('https://steemit.com/@{}/{}'.format(row.author, row.permalink))
        logger.info('Estimated Reward: {} vs. {}; Estimated votes {} vs. '
                    '{}'.format(row.predicted_reward, row.reward, row.predicted_votes, row.votes))
        logger.info('\n-------------------------------------------------'
                    '---------------------------------------------------\n')
        logger.info(row.body[:1000])
        logger.info('\n-------------------------------------------------'
                    '---------------------------------------------------\n')

    return post_frame