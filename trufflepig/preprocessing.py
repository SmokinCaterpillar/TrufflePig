import logging
import os
import multiprocessing as mp

import pandas as pd

import trufflepig.filters.stylemeasures as tfsm
import trufflepig.filters.textfilters as tftf

logger = logging.getLogger(__name__)


def filter_duplicates(frame):
    filtered = frame.drop_duplicates(subset=['author', 'permalink'],
                                     keep='last')
    if len(filtered) < len(frame):
        logger.info('Filtered {} duplicates'.format(len(frame) - len(filtered)))
    return filtered


def apply_parallel(function, iterable, ncores, chunksize=1000):
    if ncores == 1:
        return [function(x) for x in iterable]
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(ncores)

        results = [x for x in pool.map(function, iterable, chunksize)]

        pool.close()
        pool.join()

        return results


def preprocess(post_df, ncores=8, chunksize=1000,
               detect_seed=42, detect_max_length=2000,
               min_en_prob=0.8):
    logger.info('Filtering duplicates')
    post_df = filter_duplicates(post_df)

    logger.info('Filtering images')
    post_df['filtered_body'] = post_df.body.apply(lambda x:
                                                  tftf.filter_images_and_links(x))

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

    logger.info('Calculating length')
    post_df['body_length'] = post_df.filtered_body.apply(lambda x: len(x))

    large_post_df = post_df.loc[post_df.body_length >= 1000, :]
    logger.info('Keeping {} large enough posts out of {}'.format(len(large_post_df),
                                                                 len(post_df)))

    logger.info('Detecting language')
    detector = tfsm.LanguageDetector(seed=detect_seed,
                                     max_length=detect_max_length)
    large_post_df['languages'] = apply_parallel(detector.get_probabilities,
                                                      large_post_df.filtered_body,
                                                      ncores=ncores,
                                                      chunksize=chunksize)
    large_post_df['en_prob'] = large_post_df.languages.apply(lambda x:
                                                             x.get('en', 0))
    en_df = large_post_df.loc[large_post_df.en_prob >= min_en_prob, :]
    logger.info('Found {} English posts'.format(len(en_df)))

    logger.info('Splitting into sentences')
    en_df['filtered_sentences'] = en_df.filtered_body.apply(lambda x:
                                                            tfsm.split_into_sentences(x))
    en_df['num_sentences'] = en_df.filtered_sentences.apply(lambda x: len(x))

    logger.info('Computing average sentence length')
    en_df['average_sentence_length'] =  \
        en_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_average_sentence_length(x))

    logger.info('Computing sentence length variance')
    en_df['sentence_length_variance'] =  \
        en_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_sentence_length_variance(x))

    logger.info('Computing average punctuation')
    en_df['average_punctuation'] = en_df.filtered_sentences.apply(lambda x:
                                                            tfsm.compute_average_puncitation(x))

    logger.info('Combining Body and Title')
    en_df['combined'] = (en_df.title.apply(lambda x: x.lower()) + ' '
                         + en_df.filtered_body.apply(lambda x: x.lower()))

    logger.info('Filtering special characters again')
    en_df['combined'] = en_df.combined.apply(lambda x:
                                             tftf.filter_special_characters(x))

    logger.info('Filtering punctuation')
    en_df['combined'] = en_df.combined.apply(lambda x:
                                             tftf.filter_punctuation(x))

    logger.info('Replacing new lines')
    en_df['combined'] = en_df.combined.apply(lambda x: tftf.replace_newlines(x))

    logger.info('Spell checking')
    checker = tfsm.SpellErrorCounter()
    en_df['num_spelling_errors'] = apply_parallel(checker.count_mistakes,
                                              en_df.combined,
                                              ncores=ncores,
                                              chunksize=chunksize)

    logger.info('Tokenization')
    en_df['tokens'] = en_df.combined.apply(lambda x: x.split(' '))
    en_df['num_words'] = en_df.tokens.apply(lambda x: len(x))

    logger.info('Counting unique words')
    en_df['unique_words'] = en_df.tokens.apply(lambda x: len(set(x)))
    en_df['unique_ratio'] = en_df.unique_words / en_df.num_words

    logger.info('Computing characters per word')
    en_df['chars_per_word'] = en_df.body_length / en_df.num_words

    logger.info('Computing words per paragraph')
    en_df['words_per_paragraph'] = en_df.num_words / en_df.num_paragraphs

    logger.info('Computing mistakes per word')
    en_df['errors_per_word'] = en_df.num_spelling_errors / en_df.num_words

    logger.info('Counting connectors')
    en_df['num_connectors'] = en_df.tokens.apply(lambda x: tfsm.count_connectors(x))
    en_df['connectors_per_sentence'] = en_df.num_connectors / en_df.num_sentences

    final_df = en_df.dropna()
    logger.info('Final data set has {} shape'.format(final_df.shape))

    return final_df


def load_or_preprocess(post_frame, filename, *args, overwrite=False, store=True,
                       **kwargs):
    if os.path.isfile(filename) and not overwrite:
        logger.info('Found file {} will load it'.format(filename))
        post_frame = pd.read_pickle(filename, compression='gzip')
    else:
        logger.info('File {} not found, will start prepocessing'.format(filename))
        post_frame = preprocess(post_frame, *args, **kwargs)
        if store:
            directory = os.path.dirname(filename)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            logger.info('Storing file {} to disk'.format(filename))
            post_frame.to_pickle(filename, compression='gzip')
    return post_frame


