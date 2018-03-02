import logging
import os
import itertools

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
from trufflepig.preprocessing import apply_parallel
import trufflepig.filters.stylemeasures as tfsm


logger = logging.getLogger(__name__)


# List of default features used besides topics
FEATURES = ['body_length',
            'num_sentences',
            'num_paragraphs',
            'num_words',
            'num_headings',
            'unique_words',
            'unique_ratio',
            'num_spelling_errors',
            'chars_per_word',
            'words_per_paragraph',
            'errors_per_word',
            'average_sentence_length',
            'sentence_length_variance',
            'sentence_length_skew',
            'sentence_length_kurtosis',
            'average_punctuation',
            'connectors_per_sentence',
            'pronouns_per_sentence',
            'complex_word_ratio',
            'gunning_fog_index',
            'flesch_kincaid_index',
            'smog_index',
            'average_syllables',
            'syllable_variance',
            'syllable_skew',
            'syllable_kurtosis',
            'adverbs_per_sentence']

# output variables for regressor
TARGETS = ['reward', 'votes']

# Template for storing the trained model
FILENAME_TEMPLATE = 'truffle_pipeline__{time}.gz'


# tag factor punish list
PUNISH_LIST=['steem',
             'steemit',
             'crypto-news',
             'bitcoin',
             'blockchain',
             'cryptocurrency',
             'crypto',
             'dtube']


class Doc2VecModel(BaseEstimator, RegressorMixin):
    """A Doc2Vec Model following the scikit pipeline API

    NOT used in production!

    """
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
    """A K-Nearest Neighbot Doc2Vec Regressor

    Not used in production!

    """
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
    """ Gensim Latent Semantic Indexing wrapper for scikit API

    Parameters
    ----------
    no_below: int
        Filters according to minimum number of times a token must appear
    no_above: float
        Filters that a token should occur in less than `no_above` documents
    num_topics: int
        Dimensionality of topic space
    prune_at: int
        Maximum number of elements in dictionary during creation
    keep_n: int
        Maximum number of elements kept after filtering

    """
    def __init__(self, no_below, no_above, num_topics, prune_at=2000000,
                 keep_n=2000000):
        self.num_topics = num_topics
        self.no_below = no_below
        self.no_above = no_above
        self.prune_at = prune_at
        self.keep_n = keep_n
        self.lsi = None
        self.disctionary = None

    def to_corpus(self, tokens):
        """ Transfers a list of tokens into the Gensim corpus representation

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        Returns
        -------
        list of Bag of Words representations

        """
        return [self.dictionary.doc2bow(text) for text in tokens]

    def fill_dictionary(self, tokens):
        """ Fills a dictionary

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        """
        self.dictionary = corpora.Dictionary(tokens, prune_at=self.prune_at)
        self.dictionary.filter_extremes(self.no_below, self.no_above,
                                        keep_n=self.keep_n)

    def train(self, tokens):
        """ Trains the LSI model

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        """
        self.fill_dictionary(tokens)
        corpus = self.to_corpus(tokens)
        self.lsi = LsiModel(corpus, num_topics=self.num_topics)

    def project(self, tokens):
        """ Projects `tokens` into the N-dimensional LSI space

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        Returns
        -------
        lsi projection

        """
        corpus = self.to_corpus(tokens)
        return self.lsi[corpus]

    def project_dense(self, tokens):
        """ Same as `project` but returns projection as numpy array """
        projection = self.project(tokens)
        result = corpus2dense(projection, self.num_topics).T
        return result

    def fit(self, data, y=None):
        """Train in scikit API language"""
        self.train(data.tokens)
        return self

    def transform(self, data):
        """Project in scikit API language"""
        return self.project_dense(data.tokens)

    def print_topics(self, n_best=10, n_words=7):
        """ Returns a string of the best topics

        Parameters
        ----------
        n_best: int
            Number of topics to return
        n_words: int
            Number of words to show per topic

        Returns
        -------

        """
        if n_best is None:
            n_best = self.num_topics

        result = ''
        for topic in range(n_best):
            best_words = self.lsi.show_topic(topic, n_words)
            inwords = [(self.dictionary[int(x[0])], x[1]) for x in best_words]
            wordstring = ', '.join('{}: {:0.2f}'.format(*x) for x in inwords)
            result += 'Topic {}: {}\n'.format(topic, wordstring)
        return result


def create_ngrams(tokens, n):
    """Converts tokens to ngrams with white space separator

    Returns generator for n > 1
    """
    if n == 1:
        return tokens
    return (' '.join(tokens[irun:irun + n]) for irun in range(len(tokens) - n + 1))


def create_skip_bigrams(tokens, s):
    if s == -1:
        return tokens
    end = s + 2
    return (' '.join(tokens[irun:irun + end:s+1]) for irun in range(len(tokens) - end + 1))


class NGramTopicModel(TopicModel):
    """ Gensim Latent Semantic Indexing wrapper for scikit API

    Augments the standard model by increasing the token list by
    ngram tokens up to n.

    Parameters
    ----------
    no_below: int
        Filters according to minimum number of times a token must appear
    no_above: float
        Filters that a token should occur in less than `no_above` documents
    num_topics: int
        Dimensionality of topic space
    n : int
        Maximum ngram size

    """
    def __init__(self, no_below, no_above, num_topics, prune_at=5000000,
                 keep_n=400000, ngrams=(1,2)):
        super().__init__(no_below=no_below,
                         no_above=no_above,
                         num_topics=num_topics,
                         keep_n=keep_n,
                         prune_at=prune_at)
        if isinstance(ngrams, int):
            ngrams = (ngrams,)
        self.ngrams = ngrams

    def add_ngrams(self, tokens):
        """ Appends ngram tokens to the original ones

        Parameters
        ----------
        tokens list of str

        Returns
        -------
        iterator over ngrams

        """
        logger.info('Computing {} Grams'.format(self.ngrams))
        result_tokens = []
        for doc_tokens in tokens:
            ngrams = []
            for ngram in self.ngrams:
                ngrams.append(create_ngrams(doc_tokens, ngram))
            result_tokens.append(itertools.chain(*ngrams))
        return result_tokens

    def to_corpus(self, tokens):
        """ Transfers a list of tokens into the Gensim corpus representation

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        Returns
        -------
        list of Bag of Words representations

        """
        ngram_tokens = self.add_ngrams(tokens)
        return super().to_corpus(ngram_tokens)

    def fill_dictionary(self, tokens):
        """ Fills a dictionary

        Parameters
        ----------
        tokens: list of list of str
            e.g. [['hi', 'ho'], ['my', 'name', ...], ...]

        """
        ngram_tokens = self.add_ngrams(tokens)
        return super().fill_dictionary(ngram_tokens)


# class SkipBiGramTopicModel(NGramTopicModel):
#     """ Gensim Latent Semantic Indexing wrapper for scikit API
#
#     Augments the standard model by increasing the token list by
#     ngram tokens up to n.
#
#     Parameters
#     ----------
#     no_below: int
#         Filters according to minimum number of times a token must appear
#     no_above: float
#         Filters that a token should occur in less than `no_above` documents
#     num_topics: int
#         Dimensionality of topic space
#     n : int
#         Maximum ngram size
#
#     """
#     def __init__(self, no_below, no_above, num_topics, prune_at=5000000,
#                  keep_n=250000, s=2):
#         super().__init__(no_below=no_below,
#                          no_above=no_above,
#                          num_topics=num_topics,
#                          keep_n=keep_n,
#                          prune_at=prune_at,
#                          n=None)
#         self.s = s
#
#     def add_ngrams(self, tokens):
#         """ Appends ngram tokens to the original ones
#
#         Parameters
#         ----------
#         tokens list of str
#
#         Returns
#         -------
#         iterator over ngrams
#
#         """
#         logger.info('Computing Skip-N-Grams with up to {} skips'.format(self.s))
#         result_tokens = []
#         for doc_tokens in tokens:
#             ngrams = []
#             for irun in range(0, self.s + 1):
#                 ngrams.append(create_skip_bigrams(doc_tokens, irun))
#             result_tokens.append(itertools.chain(*ngrams))
#         return result_tokens


class FeatureSelector(BaseEstimator):
    """ Simply selects features from a pandas DataFrame """
    def __init__(self, features):
        self.features = features

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data.loc[:, self.features]


def create_pure_doc2vec_pipeline(knn_doc2vec_kwargs):
    """Returns a pipeline containing the KNNDoc2Vec

    NOT used in production
    """
    logger.info('Pure Doc2Vec Model')
    pipeline = Pipeline(
        steps=[('regressor', KNNDoc2Vec(**knn_doc2vec_kwargs))]
    )
    return pipeline


def create_default_pipeline(topic_kwargs, regressor_kwargs, features=FEATURES):
    """ The default pipeline used in production

    Normal features + TopicModel

    Parameters
    ----------
    topic_kwargs: dict
        Passed to the TopicModel
    regressor_kwargs: dict
        Passed to the regressor
    features: list of str
        The features taken from the preprocessed post frame

    Returns
    -------
    scikit pipeline

    """
    logger.info('Using features {}'.format(features))
    feature_generation = FeatureUnion(
        transformer_list=[
            ('feature_selection', FeatureSelector(features)),
            ('topic_model', NGramTopicModel(**topic_kwargs)),
        ]
    )

    pipeline = Pipeline(steps=[
            ('feature_generation', feature_generation),
            ('regressor', RandomForestRegressor(**regressor_kwargs))
        ]
    )

    return pipeline


def compute_log_vote_weights(target_frame):
    """ Creates training sample weights

    Weight is based on the number of votes:

        1 + np.log(1 + votes)

    Parameters
    ----------
    target_frame: DataFrame
        Frame containing the target values, must contain *votes*

    Returns
    -------
    Series of weights

    """
    logger.info('Computing sample weights')
    return 1 + np.log(1 + target_frame.votes)


def train_pipeline(post_frame, pipeline=None,
                   sample_weight_function='default', **kwargs):
    """ Trains a scikit pipeline on preprocessed posts

    Parameters
    ----------
    post_frame: DataFrame
    pipeline: scikit pipeline of None
        If None, default pipeline is created
    sample_weight_function: Func or None or 'default'
        A function that takes the target values and returns weights
        If None, no function is used
        If 'default' the log votes are used
    kwargs: **kwargs
        Passed onto the pipeline creation

    Returns
    -------
    trained scikit pipeline

    """
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
                        random_state=None,
                        **kwargs):
    """ As above but splits into training and test set and computes test score

    `train_size` is fraction of training vs test samples.

    Returns pipeline AND testing frame

    """
    train_frame, test_frame = train_test_split(post_frame,
                                               train_size=train_size,
                                               random_state=random_state)
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
    """ Runs crossvalidation on post_frame

    Parameters
    ----------
    post_frame: DataFrame
    param_grid: nested dict
        The pipeline parameters to explore
    train_size: float
        Ratio of training vs test samples
    n_jobs: int
        Number of cores for parallelization
    cv: int
        Number of crossvalidation folds
    verbose: int
        verbosity level
    n_iter: None or int
        If None than full grid search
        else n iterations of randomized grid search
    kwargs: **kwargs
        Passed onto pipeline training

    Returns
    -------
    Grid search object and testing frame

    """
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
    return grid_search, test_frame


def make_filename(current_datetime, directory):
    """Creates the filename to store models from TEMPLATE"""
    filename = FILENAME_TEMPLATE.format(time=current_datetime.strftime('%Y-%U'))
    filename = os.path.join(directory,filename)
    return filename


def model_exists(current_datetime, directory):
    """Checks if model has been stored before"""
    filename = make_filename(current_datetime, directory)
    return os.path.isfile(filename)


def load_or_train_pipeline(post_frame, directory, current_datetime=None,
                           overwrite=False, store=True, **kwargs):
    """ Loads a model or trains a new one if not found

    Parameters
    ----------
    post_frame: DataFrame
    directory: str
        Name of model directory
    current_datetime: datetime
    overwrite: bool
        If stored model should be overwritten
    store: bool
        If trained model should be stored to file
    kwargs

    Returns
    -------
    trained scikit pipeline

    """
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


def compute_tag_factor(tags, punish_list):
    """ Computes adjustment factor for tags

    Parameters
    ----------
    tags: Series
        Contains the tags of the posts

    Returns
    -------
    For each tag found in the `punish_list` the factor is multiplied by 0.85

    """
    tag_factor = tags.apply(lambda x: 1.0)
    for to_punish in punish_list:
        logger.info('...punishing {}...'.format(to_punish))
        tag_factor *= tags.apply(lambda x: 1 if to_punish not in x else 0.85)
    return tag_factor


def grammar_score_step_function(x):
    """Mapping from grammar errors per sentence to correction factor"""
    if x <= 0.05:
        return 1
    elif x <= 0.1:
        return 0.95
    elif x <= 0.2:
        return 0.9
    elif x <= 0.3:
        return 0.85
    elif x <= 0.4:
        return 0.8
    elif x <= 0.5:
        return 0.5
    elif x <=1:
        return 0.25
    else:
        return 0.1


def spelling_error_step_function(x):
    """Mapping of spelling error per word to correction factor"""
    if x <= 0.01:
        return 1.0
    elif x <= 0.02:
        return 0.95
    elif x <= 0.03:
        return 0.9
    elif x <= 0.05:
        return 0.85
    elif x <= 0.075:
        return 0.8
    else:
        return 0.7


def vote_score_step_function(x):
    """Mapping of current votes to correction factor"""
    if x >= 20:
        return 1.0
    elif x >= 10:
        return 0.95
    elif x >= 5:
        return 0.9
    else:
        return 0.4


def reward_score_step_function(x):
    """Mapping of current reward to correction factor"""
    if x >= 10:
        return 0.85
    elif x >= 1.0:
        return 1.0
    elif x >= 0.5:
        return 0.9
    else:
        return 0.4


def compute_rank_score(post_frame, punish_list=PUNISH_LIST, ncores=2, chunksize=500):
    """ Computes the ranks score too sort the truffles for the top list

    Parameters
    ----------
    post_frame: DataFrame
    punish_list: List of str
        tags that should be punished in ranking
    ncores: int
    chunksize: int

    Returns
    -------
    Series of ranks score
        The score is reward_difference * adjustment
        and adjustment=tag_factor*vote_factor*reward_factor*spelling_errors_factor*grammar_factor

    """
    logger.info('Computing tag factor...')
    tag_factor = compute_tag_factor(post_frame.tags, punish_list)

    logger.info('Computing vote factor...')
    vote_factor = post_frame.votes.apply(lambda x: vote_score_step_function(x))

    logger.info('Computing reward factor...')
    reward_factor = post_frame.reward.apply(lambda x: reward_score_step_function(x))

    logger.info('Computing spelling mistake factor...')
    spelling_errors_factor = post_frame.errors_per_word.apply(lambda x:
                                                              spelling_error_step_function(x))

    logger.info('Applying grammar check...')
    checker = tfsm.GrammarErrorCounter()
    errors_per_character = apply_parallel(checker.count_mistakes_per_character,
                                          post_frame.filtered_body,
                                          ncores=ncores,
                                          chunksize=chunksize)
    gramma_errors_per_sentence = errors_per_character * post_frame.body_length / post_frame.num_sentences
    post_frame['grammar_errors_per_sentence'] = gramma_errors_per_sentence
    grammar_factor = gramma_errors_per_sentence.apply(lambda x: grammar_score_step_function(x))

    logger.info('...Done combining reward difference and factors')
    result = post_frame.reward_difference
    final_factor = grammar_factor * reward_factor * vote_factor * tag_factor * spelling_errors_factor
    # increase negative values for low factors:
    final_factor.loc[result < 0] = 1.0 / final_factor.loc[result < 0]
    result = result * final_factor
    post_frame['rank_score'] = result
    return post_frame


def find_truffles(post_frame, pipeline, account='trufflepig',
                  punish_list=PUNISH_LIST,
                  k=25, ncores=2, chunksize=500):
    """ Digs for truffles, i.e. underpaid posts

    Filtering happens in place

    Parameters
    ----------
    post_frame: DataFrame
        Prepocessed novel data
    pipeline: trained scikit pipeline
    account: str
        The name of the bot account (it should not list itself)
    punish_list: list of str
        tags to be punished in ranking
    k: int
        Logs the first k truffles to console
    ncores: int
        For parallelization of grammar check
        More than 2 or 3 makes no sense because grammar is checked
        by LanguageTool Java Server process
    chunksize: int
        Multiprocessing chunk size

    Returns
    -------
    Sorted (from best to worst) frame of truffles with
    predicted votes and rewards

    """
    logger.info('Looking for truffles in frame of shape {} '
                'and filtering osts by '
                'myself'.format(post_frame.shape))
    to_drop = post_frame.loc[post_frame.author == account]

    post_frame.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_frame)))

    logger.info('Predicting truffles')
    predicted_rewards_and_votes = pipeline.predict(post_frame)

    post_frame['predicted_reward'] = predicted_rewards_and_votes[:, 0]
    post_frame['predicted_votes'] = predicted_rewards_and_votes[:, 1]
    post_frame['reward_difference'] = post_frame.predicted_reward - post_frame.reward

    logger.info('Computing rank score')
    post_frame = compute_rank_score(post_frame, punish_list=punish_list,
                                    ncores=ncores, chunksize=chunksize)

    post_frame.sort_values('rank_score', ascending=False, inplace=True)

    log_truffle_info(post_frame, k)

    return post_frame


def log_truffle_info(post_frame, k):
    """ Helper function to log found truffles to console """
    logger.info('\n\nDETAILED TRUFFLE INFO\n')
    for irun in range(min(k, len(post_frame))):
        row = post_frame.iloc[irun]
        truffle_str = ('\n\n----------------------------------------------'
                    '--------------------------------------------------'
                    '\n----------------------------------------------'
                    '--------------------------------------------------'
                    '\n----------------------------------------------'
                    '--------------------------------------------------'
                    '\n############ RANK {}: {} ############'.format(irun + 1 ,
                                                                     row.title))
        truffle_str += '\nhttps://steemit.com/@{}/{}'.format(row.author,
                                                        row.permalink)
        truffle_str +=('\nEstimated Reward: {:.2f} vs. {:.2f}; Estimated votes {:d} vs. '
                    '{:d} and a rank score of '
                    '{:.2f}'.format(row.predicted_reward, row.reward,
                                int(row.predicted_votes), int(row.votes),
                                row.rank_score))
        truffle_str += ('\n\n-------------------------------------------------'
                    '---------------------------------------------------\n\n')
        truffle_str += row.body[:1000]
        truffle_str += ('\n\n-------------------------------------------------'
                    '---------------------------------------------------\n')
        logger.info(truffle_str)

    logger.info('\n\nTRUFFLE SUMMARY\n')
    for irun in range(min(k, len(post_frame))):
        row = post_frame.iloc[irun]
        truffle_str = '\n\n### RANK {}: {}'.format(irun + 1 , row.title)
        truffle_str += '\nhttps://steemit.com/@{}/{}'.format(row.author,
                                                        row.permalink)
        truffle_str +=('\nEstimated Reward: {:.2f} vs. {:.2f}; Estimated votes {:d} vs. '
                    '{:d} and a rank score of '
                       '{:.2f}'.format(row.predicted_reward, row.reward,
                                int(row.predicted_votes), int(row.votes),
                                row.rank_score))
        logger.info(truffle_str)


def log_pipeline_info(pipeline,  features=FEATURES, num_topics=256):
    """Helper function to log model information to console"""
    topic_model = pipeline.named_steps['feature_generation'].transformer_list[1][1]
    logging.getLogger().info(topic_model.print_topics(n_best=None))

    feature_importance_string = 'Feature importances \n'
    feature_names = features + ['topic_{:03d}'.format(x)
                                for x in range(num_topics)]
    for kdx, importance in enumerate(pipeline.named_steps['regressor'].feature_importances_):
        name = feature_names[kdx]
        feature_importance_string += '{:03d}{:>25}: {:.3f}\n'.format(kdx, name, importance)
    logger.info(feature_importance_string)
