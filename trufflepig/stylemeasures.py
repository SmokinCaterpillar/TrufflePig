import re

import numpy as np
import langdetect
from enchant.checker import SpellChecker


def count_paragraphs(text):
    return text.count('\n\n') + 1


def detect_language(text, max_length=1024):
    """ Detexts text language, returns None in case of failure

    Parameters
    ----------
    text: str
    max_length: int
        Up to max_length characters are considered for the detection

    Returns
    -------
    str: language or None in case of failure

    """
    try:
        return langdetect.detect(text[:max_length])
    except Exception:
        return None


CAPS = "([A-Z])"
PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    """ Splits a `text` into a list of sentences.

    Taken from https://stackoverflow.com/questions/4576077/python-split-text-on-sentences

    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + CAPS + "[.] ", " \\1<prd> ", text)
    text = re.sub(ACRONYMS + " " + STARTERS, "\\1<stop> \\2", text)
    text = re.sub(CAPS + "[.]" + CAPS + "[.]" + CAPS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(CAPS + "[.]" + CAPS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + SUFFIXES + "[.] " + STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + CAPS + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def compute_average_sentence_length(text_list):
    return np.mean([len(x) for x in text_list])


def compute_sentence_length_variance(text_list):
    return np.var([len(x) for x in text_list])


class SpellErrorCounter(object):

    def __init__(self, language='en_US'):
        self.checker = SpellChecker(language)

    def count_mistakes(self, text):
        self.checker.set_text(text)
        nerrors = len([x for x in self.checker])
        return nerrors

