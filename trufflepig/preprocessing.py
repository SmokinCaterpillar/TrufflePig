import logging
import multiprocessing as mp

import pandas as pd

import trufflepig.textfilters as tftf
import trufflepig.stylemeasures as tfsm


logger = logging.getLogger(__name__)


def filter_duplicates(frame):
    filtered = frame.drop_duplicates(subset=['author', 'permalink'])
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


def preprocess(post_df, ncores=8, chunksize=100):
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
    large_post_df.loc[:, 'language'] = apply_parallel(tfsm.detect_language,
                                                      large_post_df.filtered_body,
                                                      ncores=ncores,
                                                      chunksize=chunksize)

    en_df = large_post_df.loc[large_post_df.language == 'en', :]
    logger.info('Found {} English posts'.format(len(en_df)))

    logger.info('Splitting into sentences')
    en_df['filtered_sentences'] = en_df.filtered_body.apply(lambda x:
                                                            tfsm.split_into_sentences(x))

    logger.info('Computing average sentence length')
    en_df['average_sentence_length'] =  \
        en_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_average_sentence_length(x))

    logger.info('Computing sentence length variance')
    en_df['sentence_length_variance'] =  \
        en_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_sentence_length_variance(x))

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

    final_df = en_df.dropna()
    logger.info('Final data set has {} shape'.format(final_df.shape))

    return final_df