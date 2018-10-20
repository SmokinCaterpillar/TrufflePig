import logging
import os
import multiprocessing as mp
import gc

import pandas as pd
import numpy as np
import scipy.stats as spst
from steem.amount import Amount

import trufflepig.filters.stylemeasures as tfsm
import trufflepig.filters.textfilters as tftf
import trufflepig.bchain.getaccountdata as tfga
from trufflepig.filters.blacklist import BUILD_A_WHALE_BLACKLIST


logger = logging.getLogger(__name__)


FILTER_TAGS = ('mitnebcurationtrail', 'informationwar', 'truth', 'conspiracy',
               'vaccines', 'contest', 'giveaway', 'deutsch', 'kr', 'kr-newbie',
               'nsfw', 'sex', 'daily', 'photofeed', 'gambling',
               # other weird stuff
               'steemsilvergold', 'horoscope', 'guns', 'investing', 'tib',
               # Somehow religious texts do not work in combination with others
               # maybe I need a bot just to rate spiritual content
               # for simplicity let's ignore them for now,
               # sorry, no releigious truffles in the near future!
               'bible', 'faith', 'spiritual', 'christianity', 'steemchurch',
               # Filter translations for utoptian
               'translations', 'translation')


# Stay out of the whale wars!
FILTER_AUTHORS = ('haejin', 'ew-and-patterns', 'caladium',
                  'cryptopassion', 'thirdeye7', 'shariarahammad')

# Get out plagiarismos!
FILTER_VOTERS = ('cheetah',)


def filter_duplicates(frame):
    """ Filters out duplicate entries based on author and permalink

    Filtering is inplace!

    Parameters
    ----------
    frame: DataFrame

    Returns
    -------
    DataFrame

    """
    old_len = len(frame)
    frame.drop_duplicates(subset=['author', 'permalink'],
                                     keep='last', inplace=True)
    if len(frame) < old_len:
        logger.info('Filtered {} duplicates kept {} '
                    'posts'.format(old_len - len(frame), len(frame)))
    return frame


def apply_parallel(function, iterable, ncores, chunksize=1000):
    """ Applies a `function` in parallel on `ncores`.

    Parameters
    ----------
    function: callable
    iterable: list, tuple, etc.
    ncores: int
        The number of jobs started
    chunksize: int
        Size of chunk submitted to pool

    Returns
    -------
    List of function outputs

    """
    if ncores == 1:
        return [function(x) for x in iterable]
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(ncores)

        results = [x for x in pool.imap(function, iterable, chunksize)]

        pool.close()
        pool.join()

        return results


def preprocess(post_df, ncores=4, chunksize=500,
               detect_seed=42, detect_max_length=2500,
               min_en_prob=0.9,
               min_max_body_length=(500, 35000),
               min_max_letter_ratio=(0.5, 0.85),
               min_max_num_paragraphs=(2, 250),
               min_max_num_words=(100, 12500),
               min_max_num_sentences=(5, 1250),
               min_max_words_per_paragraph=(10, 1250),
               max_errors_per_word=0.2,
               min_max_average_punctuation=(1.05, 5),
               min_max_average_sentence_length=(10, 350),
               filter_tags=FILTER_TAGS,
               filter_authors=FILTER_AUTHORS + BUILD_A_WHALE_BLACKLIST,
               filter_voters=FILTER_VOTERS,
               dropna=True):
    """ Preprocessing of raw steemit posts, filters and adds features

    All filtering happening inplace!

    Parameters
    ----------
    post_df: DataFrame
        Raw steemit posts, needs to contain
            * author
            * permalink
            * body
            * title
            * votes
            * reward
    ncores: int
        Some stuff is executed in parallel, these are the number of jobs
    chunksize: int
        Size of multiprocessing chunk
    detect_seed: int
        Seed value for language detection
    detect_max_length: int
        Maximum character size for language detection
    min_en_prob: float
        0 < min_en_prob <= 1, Minimum detection probability to classify a
        post as English
    min_max_body_length: tuple of int
        Boundaries for allowed (filtered) body length
    min_max_letter_ratio: tuple of float
        Boundaries for letters vs punctuation ratio
    min_max_num_paragraphs: tuple of int
        Boundaries for number of paragraphs
    min_max_num_words: tuple of int
        Boundaries for number of words
    min_max_num_sentences: tuple of int
        Boundaries of number of sentences
    min_max_words_per_paragraph:
        Boundaries for min max average words per paragraph
    max_errors_per_word: float
        Threshold of maximum spelling errors per word allowed
    min_max_average_punctuation: tuple of float
        Boundaries for average punctuation per sentence
    min_max_average_sentence_length: tuple of float
        Boundaries for average sentence length
    filter_tags: tuple of string
        Tags to be filtered like 'sex', 'nsfw' or controversial stuff like
        'vaccines'.
    filter_authors: tuple of string
        Authors to be filtered...
    filter_voters: tuple of string
        If vored by one of them post is excluded
    dropna: bool
        If NaN rows should be dropped

    Returns
    -------
    Filtered frame

    """
    logger.info('Filtering duplicates of {} posts'.format(len(post_df)))
    post_df = filter_duplicates(post_df)

    logger.info('Filtering authors {}'.format(filter_authors))
    filter_authors = set(filter_authors)
    author_filter = post_df.author.apply(lambda x: x in filter_authors)
    to_drop = post_df.loc[author_filter]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_df)))

    logger.info('Filtering voted by {}'.format(filter_voters))
    filter_voters = set(filter_voters)
    voted_by = post_df.active_votes.apply(lambda x: tftf.voted_by(x, filter_voters))
    to_drop = post_df.loc[voted_by]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_df)))

    logger.info('Filtering tags {}'.format(filter_tags))
    filter_tags = set(filter_tags)
    tag_filter = post_df.tags.apply(lambda x: tftf.is_in_filter_tags(x, filter_tags))
    to_drop = post_df.loc[tag_filter]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Kept {} posts'.format(len(post_df)))

    logger.info('Filtering images and links')
    post_df['filtered_body'] = post_df.body.apply(lambda x:
                                                  tftf.filter_images_and_links(x))

    logger.info('Filtering quotes')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x: tftf.filter_quotes(x))

    logger.info('Counting and filtering headings')
    post_df['num_headings'] = post_df.filtered_body.apply(lambda x:
                                                          tfsm.count_headings(x))
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_headings(x))

    logger.info('Filtering html')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x: tftf.
                                                           filter_html_tags(x))

    logger.info('Filtering urls')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_urls(x))

    logger.info('Filtering formatting')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_formatting(x))

    logger.info('Filtering special characters')
    post_df['filtered_body'] = post_df.filtered_body.apply(lambda x:
                                                           tftf.filter_special_characters(x))

    logger.info('Counting paragraphs')
    post_df['num_paragraphs'] = post_df.filtered_body.apply(lambda x:
                                                        tfsm.count_paragraphs(x))
    to_drop = post_df.loc[(post_df.num_paragraphs < min_max_num_paragraphs[0]) |
                          (post_df.num_paragraphs > min_max_num_paragraphs[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to num paragraphs limits {} '
                'kept {} posts.'.format(min_max_num_paragraphs, len(post_df)))

    logger.info('Calculating length')
    post_df['body_length'] = post_df.filtered_body.apply(lambda x: len(x))
    to_drop = post_df.loc[(post_df.body_length < min_max_body_length[0]) |
                          (post_df.body_length > min_max_body_length[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to body limits {} '
                'kept {} posts.'.format(min_max_body_length, len(post_df)))

    logger.info('Counting letters')
    post_df['letter_count'] = post_df.filtered_body.apply(lambda x: tfsm.count_letters(x))
    post_df['letter_ratio'] = post_df.letter_count / post_df.body_length
    to_drop = post_df.loc[(post_df.letter_ratio < min_max_letter_ratio[0]) |
                          (post_df.letter_ratio > min_max_letter_ratio[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to letter ratio limits {} '
                'kept {} posts.'.format(min_max_letter_ratio, len(post_df)))

    logger.info('Splitting into sentences')
    post_df['filtered_sentences'] = post_df.filtered_body.apply(lambda x:
                                                            tfsm.split_into_sentences(x))
    post_df['num_sentences'] = post_df.filtered_sentences.apply(lambda x: len(x))
    to_drop = post_df.loc[(post_df.num_sentences < min_max_num_sentences[0]) |
                          (post_df.num_sentences > min_max_num_sentences[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to num sentences limits {} '
                'kept {} posts.'.format(min_max_num_sentences, len(post_df)))

    logger.info('Computing average sentence length')
    post_df['average_sentence_length'] =  post_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_average_sentence_length(x))
    to_drop = post_df.loc[(post_df.average_sentence_length < min_max_average_sentence_length[0]) |
                          (post_df.average_sentence_length > min_max_average_sentence_length[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to avg. sentences limits {} '
                'kept {} posts.'.format(min_max_average_sentence_length, len(post_df)))

    logger.info('Intermediate garbage collection.')
    gc.collect()

    logger.info('Computing sentence length variance')
    post_df['sentence_length_variance'] = post_df.filtered_sentences.apply(lambda x:
                                          tfsm.compute_sentence_length_variance(x))

    logger.info('Computing sentence length skew')
    post_df['sentence_length_skew'] = post_df.filtered_sentences.apply(lambda x:
                                          tfsm.compute_sentence_length_skew(x))

    logger.info('Computing sentence length kurtosis')
    post_df['sentence_length_kurtosis'] = post_df.filtered_sentences.apply(lambda x:
                                          tfsm.compute_sentence_length_kurtosis(x))

    logger.info('Combining Body and Title')
    post_df['combined'] = (post_df.title.apply(lambda x: x.lower()) + ' '
                         + post_df.filtered_body.apply(lambda x: x.lower()))

    logger.info('Filtering special characters again')
    post_df['combined'] = post_df.combined.apply(lambda x:
                                             tftf.filter_special_characters(x))

    logger.info('Filtering punctuation')
    post_df['combined'] = post_df.combined.apply(lambda x:
                                             tftf.filter_punctuation(x))

    logger.info('Replacing new lines')
    post_df['combined'] = post_df.combined.apply(lambda x: tftf.replace_newlines(x))

    logger.info('Computing average punctuation')
    post_df['average_punctuation'] = post_df.filtered_sentences.apply(lambda x:
                                                            tfsm.compute_average_puncitation(x))
    to_drop = post_df.loc[(post_df.average_punctuation < min_max_average_punctuation[0]) |
                          (post_df.average_punctuation > min_max_average_punctuation[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to punctuation limits {} '
                'kept {} posts.'.format(min_max_average_punctuation, len(post_df)))

    post_df.drop('filtered_sentences', axis=1, inplace=True)
    logger.info('Intermediate garbage collection.')
    gc.collect()

    logger.info('Detecting language')
    detector = tfsm.LanguageDetector(seed=detect_seed,
                                     max_length=detect_max_length)
    post_df['languages'] = apply_parallel(detector.get_probabilities,
                                                      post_df.filtered_body,
                                                      ncores=ncores,
                                                      chunksize=chunksize)
    post_df['en_prob'] = post_df.languages.apply(lambda x: x.get('en', 0))
    to_drop = post_df.loc[post_df.en_prob < min_en_prob]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Found {} English posts with threshold {}'.format(len(post_df),
                                                                  min_en_prob))

    logger.info('Spell checking')
    checker = tfsm.SpellErrorCounter()
    post_df['num_spelling_errors'] = apply_parallel(checker.count_mistakes,
                                              post_df.combined,
                                              ncores=ncores,
                                              chunksize=chunksize)

    logger.info('Tokenization')
    post_df['tokens'] = post_df.combined.apply(lambda x: x.split(' '))
    post_df.drop('combined', axis=1, inplace=True)
    logger.info('Intermediate garbage collection.')
    gc.collect()

    logger.info('Computing number of words')
    post_df['num_words'] = post_df.tokens.apply(lambda x: len(x))
    to_drop = post_df.loc[(post_df.num_words < min_max_num_words[0]) |
                          (post_df.num_words > min_max_num_words[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to num words limits {} '
                'kept {} posts.'.format(min_max_num_words, len(post_df)))

    logger.info('Computing words per paragraph')
    post_df['words_per_paragraph'] = post_df.num_words / post_df.num_paragraphs
    to_drop = post_df.loc[(post_df.words_per_paragraph < min_max_words_per_paragraph[0]) |
                          (post_df.words_per_paragraph > min_max_words_per_paragraph[1])]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to num words per paragraph limits {} '
                'kept {} posts.'.format(min_max_words_per_paragraph, len(post_df)))

    logger.info('Computing mistakes per word')
    post_df['errors_per_word'] = post_df.num_spelling_errors / post_df.num_words
    to_drop = post_df.loc[post_df.errors_per_word > max_errors_per_word]
    post_df.drop(to_drop.index, inplace=True)
    logger.info('Filtered according to spelling mistake limit {} per word '
                'kept {} posts.'.format(max_errors_per_word, len(post_df)))

    logger.info('Intermediate garbage collection.')
    gc.collect()

    logger.info('Counting unique words')
    post_df['unique_words'] = post_df.tokens.apply(lambda x: len(set(x)))
    post_df['unique_ratio'] = post_df.unique_words / post_df.num_words

    logger.info('Computing characters and characters per word')
    post_df['num_chars'] = post_df.tokens.apply(lambda x: tfsm.count_characters(x))
    post_df['chars_per_word'] = post_df.num_chars / post_df.num_words

    logger.info('Counting connectors')
    post_df['num_connectors'] = post_df.tokens.apply(lambda x: tfsm.count_connectors(x))
    post_df['connectors_per_sentence'] = post_df.num_connectors / post_df.num_sentences

    logger.info('Counting pronouns')
    post_df['num_pronouns'] = post_df.tokens.apply(lambda x: tfsm.count_pronouns(x))
    post_df['pronouns_per_sentence'] = post_df.num_pronouns / post_df.num_sentences

    logger.info('Counting adverbs')
    post_df['num_adverbs'] = post_df.tokens.apply(lambda x: tfsm.adverb_estimate(x))
    post_df['adverbs_per_sentence'] = post_df.num_adverbs / post_df.num_sentences

    logger.info('Calculating syllables')
    syllable_converter = tfsm.SyllableConverter()
    post_df['token_syllables'] = post_df.tokens.apply(lambda x:
                                                  syllable_converter.tokens2syllablses(x))
    logger.info('Computing features based on the syllables')
    post_df['num_syllables'] = post_df.token_syllables.apply(lambda x: sum(x))
    post_df['num_complex_words'] = post_df.token_syllables.apply(lambda x:
                                                             sum([y >= 3 for y in x]))
    post_df['complex_word_ratio'] = post_df.num_complex_words / post_df.num_words
    post_df['average_syllables'] = post_df.token_syllables.apply(lambda x: np.mean(x))
    post_df['syllable_variance'] = post_df.token_syllables.apply(lambda x: np.var(x))
    post_df['syllable_skew'] = post_df.token_syllables.apply(lambda x: spst.skew(x))
    post_df['syllable_kurtosis'] = post_df.token_syllables.apply(lambda x: spst.kurtosis(x))

    logger.info('Computing readability indices')
    post_df['gunning_fog_index'] = tfsm.gunning_fog_index(num_words=post_df.num_words,
                                                        num_complex_words=post_df.num_complex_words,
                                                        num_sentences=post_df.num_sentences)
    post_df['flesch_kincaid_index'] = tfsm.flesch_kincaid_index(num_syllables=post_df.num_syllables,
                                                              num_words=post_df.num_words,
                                                              num_sentences=post_df.num_sentences)
    post_df['smog_index']= tfsm.smog_index(num_complex_words=post_df.num_complex_words,
                                         num_sentences=post_df.num_sentences)
    post_df['automated_readability_index'] = tfsm.automated_readability_index(num_chars=post_df.num_chars,
                                                                        num_words=post_df.num_words,
                                                                        num_sentences=post_df.num_sentences)
    post_df['coleman_liau_index'] = tfsm.coleman_liau_index(num_chars=post_df.num_chars,
                                                            num_words=post_df.num_words,
                                                            num_sentences=post_df.num_sentences)

    if dropna:
        logger.info('Dropping NaN rows')
        post_df.dropna(inplace=True)
    logger.info('Final data set has {} shape'.format(post_df.shape))

    return post_df


def load_or_preprocess(post_frame, filename, *args,
                       overwrite=False, store=True,
                       steem_args_for_upvote=None,
                       bots=tfga.BITBOTS,
                       **kwargs):
    """ Tries to load a preprocessed frame if not found preprocessing starts.

    Parameters
    ----------
    post_frame: DataFrame
    filename: str
        Filename of data to load
    args: *args
        Arguments passed to normal preprocessing
    steem_args_for_upvote: dict
        Steem arguments, leave None to not load corrections
    overwrite: bool
        If preprocessing should be started and overwrite existing file
    store: bool
        If preprocessed frame should be stored to file
    kwargs: **kwargs
        Arguments passed to preprocessing

    Returns
    -------
    DataFrame

    """
    if os.path.isfile(filename) and not overwrite:
        logger.info('Found file {} will load it'.format(filename))
        post_frame = pd.read_pickle(filename, compression='gzip')
    else:
        logger.info('File {} not found, will start prepocessing'.format(filename))
        post_frame = preprocess(post_frame, *args, **kwargs)
        if steem_args_for_upvote:
            logger.info('Looking for bought upvotes!')
            min_datetime = post_frame.created.min()
            max_datetime = post_frame.created.max() + pd.Timedelta(days=8)
            upvote_payments, _ = tfga.get_upvote_payments_to_bots(steem_args_for_upvote,
                                                                min_datetime=min_datetime,
                                                                max_datetime=max_datetime,
                                                                bots=bots)
            post_frame = compute_bidbot_correction(post_frame,
                                                   upvote_payments)
        if store:
            directory = os.path.dirname(filename)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            logger.info('Storing file {} to disk'.format(filename))
            post_frame.to_pickle(filename, compression='gzip')
    return post_frame


def compute_bidbot_correction(post_frame, upvote_payments, sbd_punishment_factor=2.2,
                              steem_punishment_factor=2.0):
    post_frame['sbd_bought_reward'] = 0.
    post_frame['steem_bought_reward'] = 0.
    post_frame['bought_votes'] = 0

    post_frame.set_index(['author', 'permalink'], inplace=True)

    for (author, permalink), payments in upvote_payments.items():
        if (author, permalink) in post_frame.index:
            sbd = 0
            steem = 0
            votes = 0
            for payment in payments.values():
                amount = Amount(payment['amount'])
                value = amount.amount
                asset = amount.asset
                votes += 1
                if asset == 'SBD':
                    sbd += value
                elif asset == 'STEEM':
                    steem += value
                else:
                    raise RuntimeError('W00t?')
            post_frame.loc[(author, permalink),
                       ['sbd_bought_reward',
                        'steem_bought_reward',
                        'bought_votes']] = sbd, steem, votes

    post_frame.reset_index(inplace=True)
    post_frame['adjusted_reward'] = post_frame.reward - \
                                    post_frame.sbd_bought_reward * sbd_punishment_factor - \
                                    post_frame.steem_bought_reward * steem_punishment_factor
    post_frame.loc[post_frame.adjusted_reward < 0, 'adjusted_reward'] = 0
    post_frame['adjusted_votes'] = post_frame.votes - post_frame.bought_votes
    post_frame.loc[post_frame.adjusted_votes < 0, 'adjusted_votes'] = 0

    num_articles = (post_frame.bought_votes > 0).sum()
    percent = num_articles / len(post_frame) * 100
    total_steem = post_frame.steem_bought_reward.sum()
    total_sbd = post_frame.sbd_bought_reward.sum()
    total_votes = post_frame.bought_votes.sum()
    logger.info('Found {} upvoted articles ({:.2f}%) with '
                'total {:.3f} STEEM {:.3f} SBD, and {} bought votes!'.format(num_articles,
                                                                       percent,
                                                                       total_steem,
                                                                       total_sbd,
                                                                       total_votes))
    return post_frame
