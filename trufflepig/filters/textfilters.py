import re
import logging


logger = logging.getLogger(__name__)


def filter_html_tags(text):
    return re.sub('</?[a-z]{1,11}>', '', text)


def filter_images_and_links(text):
    return re.sub('!?\[[-a-zA-Z0-9?@: %._\+~#=/()]*\]\([-a-zA-Z0-9?@:%._\+~#=/()]+\)', '', text)


def filter_urls(text):
    return re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]'
                   '[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.'
                   '[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}'
                   '|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]'
                   '{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})', '', text)


def filter_special_characters(text):
    return re.sub('[^A-Za-z0-9\s;,.?!]+', '', text)


def filter_formatting(text):
    text = re.sub('&?nbsp', ' ',text)
    text = re.sub('aligncenter', '', text)
    text = re.sub('styletextalign', '', text)
    text = re.sub('href', '', text)
    text = re.sub('h[1-5]', '', text)
    return text


def replace_newlines(text):
    return re.sub('\s+', ' ', text)


def filter_punctuation(text):
    return re.sub('[;,.?!]+', '', text)


def is_in_filter_tags(tags, filter_tags):
    try:
        return bool(set(tags).intersection(filter_tags))
    except TypeError:
        logger.exception('Could not identify tags {}, '
                         'will return True anyway'.format(tags))
        return True
