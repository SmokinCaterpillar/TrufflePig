import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, \
    RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

from gensim.models.lsimodel import LsiModel
from gensim import corpora
from gensim.matutils import corpus2dense
import gensim.models.doc2vec as d2v

from trufflepig.utils import progressbar


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
            'grammar_errors_per_sentence',
            'average_sentence_length',
            'sentence_length_variance',
            'average_punctuation',
            'connectors_per_sentence',
            'pronouns_per_sentence',
            'complex_word_ratio',
            'gunning_fog_index',
            'flesch_kincaid_index',
            'smog_index',
            'average_syllables',
            'syllable_variance',
            'adverbs_per_sentence']

TARGETS = ['reward', 'votes']

FILENAME_TEMPLATE = 'truffle_pipeline__{time}.gz'


class Doc2VecModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.25, min_alpha=0.01, size=32,
                 window=8, min_count=5, workers=4, sample=1e-4,
                 negative=5, epochs=5, infer_steps=10):
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sample = sample
        self.negative = negative
        self.epochs=epochs
        self.model = None
        self.infer_steps = infer_steps


    def create_tagged_documents(self, document_frame):
        permalinks = document_frame.permalink
        authors = document_frame.author
        tokens = document_frame.tokens
        return [d2v.TaggedDocument(words=tks, tags=[author+'/'+permalink])
                for tks, author, permalink in zip(tokens, authors, permalinks)]

    def train(self, document_frame):
        tagged_docs = self.create_tagged_documents(document_frame)
        model = d2v.Doc2Vec(alpha=self.alpha, min_alpha=self.min_alpha,
                            size=self.size, window=self.window,
                            min_count=self.min_count, workers=self.workers,
                            sample=self.sample, negative=self.negative)
        logger.info('Building vocab')
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count,
                    epochs = self.epochs)
        logger.info('Training successfull deleting temporary data..')
        model.delete_temporary_training_data()
        self.model = model
        logger.info('...done, have fun with your new network!')

    def fit(self, document_frame, y=None):
        self.train(document_frame)
        return self

    def transform(self, document_frame):
        dim = self.model.vector_size
        inputs = np.zeros((len(document_frame), dim))
        logger.info('Transforming documents into matrix of '
                    'shape {}'.format(inputs.shape))
        tagged_docs = self.create_tagged_documents(document_frame)
        for kdx, (author, permalink) in enumerate(zip(document_frame.author,
                                                      document_frame.permalink)):
            try:
                inputs[kdx, :] = self.model.docvecs[author+'/'+permalink]
            except KeyError:
                # infer the test vector
                inputs[kdx, :] = self.model.infer_vector(tagged_docs[kdx].words,
                                                         steps=self.infer_steps)
            progressbar(kdx, len(inputs), logger=logger)
        return inputs


class KNNDoc2Vec(Doc2VecModel):
    def __init__(self, knn=5, alpha=0.25, min_alpha=0.01, size=32,
                 window=8, min_count=5, workers=4, sample=1e-4,
                 negative=5, epochs=5, infer_steps=10):
        super().__init__(alpha=alpha, min_alpha=min_alpha, size=size,
                         window=window, min_count=min_count, workers=workers,
                         sample=sample, negative=negative, epochs=epochs,
                         infer_steps=infer_steps)
        self.knn = knn
        self.trainY = None

    def fit(self, document_frame, target_frame, sample_weight=None):
        self.trainY = target_frame.copy()
        self.trainY['doctag'] = document_frame.author + '/' + document_frame.permalink
        self.trainY = self.trainY.set_index('doctag')
        return super().fit(document_frame, target_frame)

    def predict(self, document_frame):
        logger.info('Predicting {} values'.format(len(document_frame)))
        values = self.transform(document_frame)
        results = np.zeros((len(values), self.trainY.shape[1]))
        logger.info('Finding {} nearest neighbors'.format(self.knn))
        for idx in range(len(values)):
            vector = values[idx, :]
            returns = self.model.docvecs.most_similar(positive=[vector], topn=self.knn)
            indices = [doctag for doctag, sim in returns]
            mean_vals = self.trainY.loc[indices, :].mean()
            results[idx, :] = mean_vals
            progressbar(idx, len(values), logger=logger)
        return results


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
            best_words = self.lsi.show_topic(topic, n_words)
            inwords = [(self.dictionary[int(x[0])], x[1]) for x in best_words]
            wordstring = ', '.join('{}: {:0.2f}'.format(*x) for x in inwords)
            result += 'Topic {}: {}\n'.format(topic, wordstring)
        return result


class FeatureSelector(BaseEstimator):
    def __init__(self, features):
        self.features = features
        #self.scaler = StandardScaler()

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data.loc[:, self.features]


def create_pure_doc2vec_pipeline(knn_doc2vec_kwargs):
    logger.info('Pure Doc2Vec Model')
    pipeline = Pipeline(
        steps=[('regressor', KNNDoc2Vec(**knn_doc2vec_kwargs))]
    )
    return pipeline


def create_default_pipeline(topic_kwargs, regressor_kwargs, features=FEATURES):
    logger.info('Using features {}'.format(features))
    feature_generation = FeatureUnion(
        transformer_list=[
            ('feature_selection', FeatureSelector(features)),
            ('topic_model', TopicModel(**topic_kwargs)),
        ]
    )

    pipeline = Pipeline(steps=[
            ('feature_generation', feature_generation),
            ('regressor', RandomForestRegressor(**regressor_kwargs))
        ]
    )

    return pipeline


def compute_log_vote_weights(target_frame):
    logger.info('Computing sample weights')
    return 1 + np.log(1 + target_frame.votes)


def train_pipeline(post_frame, pipeline=None, sample_weight_function='default', **kwargs):
    targets = kwargs.pop('targets', TARGETS)

    logger.info('Training pipeline with targets {} and {} '
                'samples...'.format(targets, len(post_frame)))
    target_frame = post_frame.loc[:, targets]

    if sample_weight_function == 'default':
        sample_weight_function = compute_log_vote_weights

    if sample_weight_function is not None:
        sample_weight = sample_weight_function(target_frame)
    else:
        sample_weight = None

    if pipeline is None:
        logger.info('...Creating the default pipeline...')
        pipeline = create_default_pipeline(**kwargs)

    pipeline.fit(post_frame, target_frame,
                 regressor__sample_weight=sample_weight)

    score = pipeline.score(post_frame, target_frame,
                           sample_weight=sample_weight)
    logger.info('...Done! Training score {}'.format(score))

    return pipeline


def train_test_pipeline(post_frame, pipeline=None,
                        train_size=0.8, sample_weight_function='default',
                        **kwargs):
    train_frame, test_frame = train_test_split(post_frame,
                                               train_size=train_size)
    targets = kwargs.get('targets', TARGETS)
    pipeline = train_pipeline(train_frame, pipeline=pipeline,
                              sample_weight_function=sample_weight_function,
                              **kwargs)

    target_frame = test_frame.loc[:, targets]

    logger.info('Using test data...')

    if sample_weight_function == 'default':
        sample_weight_function = compute_log_vote_weights

    if sample_weight_function is not None:
        sample_weight = sample_weight_function(target_frame)
    else:
        sample_weight = None

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

    pipeline = create_default_pipeline(**kwargs)
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

    logger.info("FINAL TEST Score {} \n of best estimator:\n\n{}".format(score,
                                        best_estimator.get_params()))


def make_filename(current_datetime, directory):
    filename = FILENAME_TEMPLATE.format(time=current_datetime.strftime('%Y-%U'))

    filename = os.path.join(directory,filename)
    return filename


def model_exists(current_datetime, directory):
    filename = make_filename(current_datetime, directory)
    return os.path.isfile(filename)


def load_or_train_pipeline(post_frame, directory, current_datetime=None,
                           overwrite=False, store=True, **kwargs):
    if current_datetime is None:
        current_datetime = pd.datetime.utcnow()
    else:
        current_datetime = pd.to_datetime(current_datetime)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    filename = make_filename(current_datetime, directory)

    if model_exists(current_datetime, directory) and not overwrite:
        logger.info('Found file {} will load it'.format(filename))
        pipeline = joblib.load(filename)
    else:
        logger.info('File {} not found, will start training'.format(filename))
        pipeline = train_pipeline(post_frame, **kwargs)
        if store:
            logger.info('Storing file {} to disk'.format(filename))
            joblib.dump(pipeline, filename, compress=8)
    return pipeline


def find_truffles(post_frame, pipeline, min_max_reward=(1.0, 10), min_votes=5, k=10):
    logger.info('Looking for truffles and filtering preprocessed data further. '
                'min max reward {} and min votes {}'.format(min_max_reward, min_votes))
    to_drop = post_frame.loc[(post_frame.reward < min_max_reward[0]) |
                                (post_frame.reward > min_max_reward[1]) |
                                (post_frame.votes < min_votes)]

    post_frame.drop(to_drop.index, inplace=True)

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
                    '\n----------------------------------------------'
                    '--------------------------------------------------'
                    '\n----------------------------------------------'
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


def log_pipeline_info(pipeline):
    topic_model = pipeline.named_steps['feature_generation'].transformer_list[1][1]
    logging.getLogger().info(topic_model.print_topics(n_best=None))

    feature_importance_string = 'Feature importances \n'
    for kdx, importance in enumerate(pipeline.named_steps['regressor'].feature_importances_):
        feature_importance_string += '{:03d}: {:.3f}\n'.format(kdx, importance)
    logger.info(feature_importance_string)
