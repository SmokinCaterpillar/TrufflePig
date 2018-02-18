import logging
import os
import multiprocessing as mp

import pandas as pd
import numpy as np

import trufflepig.filters.stylemeasures as tfsm
import trufflepig.filters.textfilters as tftf

logger = logging.getLogger(__name__)


FILTER_TAGS = ('mitnebcurationtrail', 'informationwar', 'truth', 'conspiracy',
               'vaccines', 'contest', 'giveaway', 'deutsch', 'kr', 'kr-newbie')


def filter_duplicates(frame):
    old_len = len(frame)
    frame.drop_duplicates(subset=['author', 'permalink'],
                                     keep='last', inplace=True)
    if len(frame) < old_len:
        logger.info('Filtered {} duplicates kept {} '
                    'posts'.format(old_len - len(frame), len(frame)))
    return frame


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
               min_en_prob=0.9,
               min_max_body_length=(500, 25000),
               min_max_letter_ratio=(0.5, 0.85),
               min_max_num_paragraphs=(2, 100),
               min_max_num_words=(100, 7500),
               min_max_num_sentences=(5, 750),
               min_max_words_per_paragraph=(20, 500),
               max_erros_per_word=0.1,
               min_max_average_punctuation=(1.05, 5),
               min_max_average_sentence_length=(10, 300),
               filter_tags = FILTER_TAGS,
               ):
    logger.info('Filtering duplicates of {} posts'.format(len(post_df)))
    post_df = filter_duplicates(post_df)

    logger.info('Filtering dodgy tags {}'.format(filter_tags))
    filter_tags = set(filter_tags)
    tag_filter = post_df.tags.apply(lambda x: tftf.is_in_filter_tags(x, filter_tags))
    post_df = post_df.loc[~tag_filter]
    logger.info('Kept {} posts'.format(len(post_df)))

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
    post_df = post_df.loc[(post_df.num_paragraphs >= min_max_num_paragraphs[0]) &
                          (post_df.num_paragraphs <= min_max_num_paragraphs[1])]
    logger.info('Filtered according to num paragraphs limits {} '
                'kept {} posts.'.format(min_max_num_paragraphs, len(post_df)))

    logger.info('Calculating length')
    post_df['body_length'] = post_df.filtered_body.apply(lambda x: len(x))
    post_df = post_df.loc[(post_df.body_length >= min_max_body_length[0]) &
                          (post_df.body_length <= min_max_body_length[1]), :]
    logger.info('Filtered according to body limits {} '
                'kept {} posts.'.format(min_max_body_length, len(post_df)))

    logger.info('Counting letters')
    post_df['letter_count'] = post_df.filtered_body.apply(lambda x:
                                                                      tfsm.count_letters(x))
    post_df['letter_ratio'] = post_df.letter_count / post_df.body_length
    post_df = post_df.loc[(post_df.letter_ratio >= min_max_letter_ratio[0]) &
                          (post_df.letter_ratio <= min_max_letter_ratio[1])]
    logger.info('Filtered according to letter ratio limits {} '
                'kept {} posts.'.format(min_max_letter_ratio, len(post_df)))

    logger.info('Splitting into sentences')
    post_df['filtered_sentences'] = post_df.filtered_body.apply(lambda x:
                                                            tfsm.split_into_sentences(x))
    post_df['num_sentences'] = post_df.filtered_sentences.apply(lambda x: len(x))
    post_df = post_df.loc[(post_df.num_sentences >= min_max_num_sentences[0]) &
                          (post_df.num_sentences <= min_max_num_sentences[1])]
    logger.info('Filtered according to num sentences limits {} '
                'kept {} posts.'.format(min_max_num_sentences, len(post_df)))

    logger.info('Computing average sentence length')
    post_df['average_sentence_length'] =  \
        post_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_average_sentence_length(x))
    post_df = post_df.loc[(post_df.average_sentence_length >= min_max_average_sentence_length[0]) &
                          (post_df.average_sentence_length <= min_max_average_sentence_length[1])]
    logger.info('Filtered according to avg. sentences limits {} '
                'kept {} posts.'.format(min_max_average_sentence_length, len(post_df)))

    logger.info('Computing sentence length variance')
    post_df['sentence_length_variance'] =  \
        post_df.filtered_sentences.apply(lambda x:
                                       tfsm.compute_sentence_length_variance(x))

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

    logger.info('Tokenization')
    post_df['tokens'] = post_df.combined.apply(lambda x: x.split(' '))
    post_df['num_words'] = post_df.tokens.apply(lambda x: len(x))
    post_df = post_df.loc[(post_df.num_words >= min_max_num_words[0]) &
                          (post_df.num_words <= min_max_num_words[1])]
    logger.info('Filtered according to num words limits {} '
                'kept {} posts.'.format(min_max_num_words, len(post_df)))

    logger.info('Counting unique words')
    post_df['unique_words'] = post_df.tokens.apply(lambda x: len(set(x)))
    post_df['unique_ratio'] = post_df.unique_words / post_df.num_words

    logger.info('Computing characters per word')
    post_df['chars_per_word'] = post_df.body_length / post_df.num_words

    logger.info('Computing words per paragraph')
    post_df['words_per_paragraph'] = post_df.num_words / post_df.num_paragraphs
    post_df = post_df.loc[(post_df.words_per_paragraph >= min_max_words_per_paragraph[0]) &
                          (post_df.words_per_paragraph <= min_max_words_per_paragraph[1])]
    logger.info('Filtered according to num words per paragraph limits {} '
                'kept {} posts.'.format(min_max_words_per_paragraph, len(post_df)))

    logger.info('Computing average punctuation')
    post_df['average_punctuation'] = post_df.filtered_sentences.apply(lambda x:
                                                            tfsm.compute_average_puncitation(x))
    post_df = post_df.loc[(post_df.average_punctuation >= min_max_average_punctuation[0]) &
                          (post_df.average_punctuation <= min_max_average_punctuation[1])]
    logger.info('Filtered according to punctuation limits {} '
                'kept {} posts.'.format(min_max_average_punctuation, len(post_df)))

    logger.info('Detecting language')
    detector = tfsm.LanguageDetector(seed=detect_seed,
                                     max_length=detect_max_length)
    post_df['languages'] = apply_parallel(detector.get_probabilities,
                                                      post_df.filtered_body,
                                                      ncores=ncores,
                                                      chunksize=chunksize)
    post_df['en_prob'] = post_df.languages.apply(lambda x: x.get('en', 0))
    en_df = post_df.loc[post_df.en_prob >= min_en_prob, :]
    logger.info('Found {} English posts with threshold {}'.format(len(en_df),
                                                                  min_en_prob))

    logger.info('Counting connectors')
    en_df['num_connectors'] = en_df.tokens.apply(lambda x: tfsm.count_connectors(x))
    en_df['connectors_per_sentence'] = en_df.num_connectors / en_df.num_sentences

    logger.info('Counting pronouns')
    en_df['num_pronouns'] = en_df.tokens.apply(lambda x: tfsm.count_pronouns(x))
    en_df['pronouns_per_sentence'] = en_df.num_pronouns / en_df.num_sentences

    logger.info('Spell checking')
    checker = tfsm.SpellErrorCounter()
    en_df['num_spelling_errors'] = apply_parallel(checker.count_mistakes,
                                              en_df.combined,
                                              ncores=ncores,
                                              chunksize=chunksize)

    logger.info('Computing mistakes per word')
    en_df['errors_per_word'] = en_df.num_spelling_errors / en_df.num_words
    en_df = en_df.loc[en_df.errors_per_word <= max_erros_per_word]
    logger.info('Filtered according to spelling mistake limit {} per word'
                'kept {} posts.'.format(max_erros_per_word, len(en_df)))

    logger.info('Counting adverbs')
    en_df['num_adverbs'] = en_df.tokens.apply(lambda x: tfsm.adverb_estimate(x))
    en_df['adverbs_per_sentence'] = en_df.num_adverbs / en_df.num_sentences

    logger.info('Calculating syllables')
    syllable_converter = tfsm.SyllableConverter()
    en_df['token_syllables'] = en_df.tokens.apply(lambda x:
                                                  syllable_converter.tokens2syllablses(x))
    logger.info('Computing features based on the syllables')
    en_df['num_syllables'] = en_df.token_syllables.apply(lambda x: sum(x))
    en_df['num_complex_words'] = en_df.token_syllables.apply(lambda x:
                                                             sum([y >= 3 for y in x]))
    en_df['complex_word_ratio'] = en_df.num_complex_words / en_df.num_words
    en_df['average_syllables'] = en_df.token_syllables.apply(lambda x: np.mean(x))
    en_df['syllable_variance'] = en_df.token_syllables.apply(lambda x: np.var(x))
    en_df['gunning_fog_index'] = tfsm.gunning_fog_index(num_words=en_df.num_words,
                                                        num_complex_words=en_df.num_complex_words,
                                                        num_sentences=en_df.num_sentences)
    en_df['flesch_kincaid_index'] = tfsm.flesch_kincaid_index(num_syllables=en_df.num_syllables,
                                                              num_words=en_df.num_words,
                                                              num_sentences=en_df.num_sentences)
    en_df['smog_index']= tfsm.smog_index(num_complex_words=en_df.num_complex_words,
                                         num_sentences=en_df.num_sentences)

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


