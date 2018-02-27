import re
import string

import numpy as np
import langdetect
import language_check
import pyphen
from enchant.checker import SpellChecker

CAPS = "([A-Z])"
PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov)"

CONNECTORS = {
    'however',
    'nevertheless',
    'nonetheless',
    'yet',
    'instead',
    'contrary',
    'likewise',
    'similarly',
    'correspondingly',
    'accordingly',
    'consequently',
    'hence',
    'thus',
    'besides',
    'furthermore',
    'moreover',
    'finally',
    'firstly',
    'secondly',
    'thirdly',
    'specifically',
    'instance',
    'indeed',
    'regardimg',
    'conclusion',
    'rather',
    'afterwards',
}

PRONOUNS = {'i', 'me', 'my' , 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'it', 'its', 'we', 'us' 'our', 'they', 'them', 'their'}

STEEMIT_WORDS = (
        'crypto',
        'cryptocurrency',
        'cryptocurrencies',
        'bitcoin',
        'blockchain',
        'litecoin',
        'ethereum',
        'ripple',
        'steem',
        'steemit',
        'resteem',
        'upvote',
        'downvote',
)

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


def count_headings(text):
    return len(re.findall('(\n|^)(#{1,6}| *<h[1-6]>)[^\n#]*\n', text))


def count_paragraphs(text):
    return text.count('\n\n') + 1


def compute_average_sentence_length(text_list):
    return np.mean([len(x) for x in text_list])


def compute_sentence_length_variance(text_list):
    return np.var([len(x) for x in text_list])


def compute_average_puncitation(text_list):
    punctuation = [sum(1 for x in sentence if x in set(string.punctuation) )
                         for sentence in text_list]
    return np.mean(punctuation)


def count_connectors(tokens):
    return sum(1 for x in tokens if x in CONNECTORS)


def count_pronouns(tokens):
    return sum(1 for x in tokens if x in PRONOUNS)


def count_letters(text):
    return len(re.findall('[a-zA-Z]', text))


def gunning_fog_index(num_words, num_complex_words, num_sentences):
    """https://en.wikipedia.org/wiki/Gunning_fog_index"""
    return 0.4*((num_words / num_sentences) + 100 * (num_complex_words / num_words))


def flesch_kincaid_index(num_syllables, num_words, num_sentences):
    """https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests"""
    return 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)


def smog_index(num_complex_words, num_sentences):
    return 1.0430 * np.sqrt(num_complex_words * 30 / num_sentences) + 3.1291


def adverb_estimate(tokens):
    return sum([1 for x in tokens if x.endswith('ly')])


class SyllableConverter(object):
    def __init__(self, language='en'):
        self.dic = pyphen.Pyphen(lang=language)

    def tokens2syllablses(self, tokens):
        results = []
        for token in tokens:
            nsyllables = len(self.dic.positions(token)) + 1
            results.append(nsyllables)
        return results


class SpellErrorCounter(object):

    def __init__(self, language='en_US', pwl=STEEMIT_WORDS):
        self.checker = SpellChecker(language)
        for word in pwl:
            self.checker.add(word)

    def count_mistakes(self, text):
        self.checker.set_text(text)
        nerrors = len([x for x in self.checker])
        return nerrors


class GrammarErrorCounter(object):
    def __init__(self, language='en-US', max_length=5000):
        self.max_length = max_length
        self.tool = language_check.LanguageTool(language=language)
        self.tool.disable_spellchecking()

    def count_mistakes_per_character(self, text):
        if self.max_length:
            text = text[:self.max_length]
        nerrors = len(self.tool.check(text))
        return nerrors / len(text)


class LanguageDetector(object):
    def __init__(self, max_length=5000, seed=42):
        self.max_length = max_length
        self.factory = langdetect.DetectorFactory()
        self.factory.set_seed(seed)
        self.factory.load_profile(langdetect.PROFILES_DIRECTORY)

    def detect_language(self, text):
        try:
            detector = self.factory.create()
            if self.max_length:
                text = text[:self.max_length]
            detector.append(text)
            return detector.detect()
        except Exception as e:
            return None

    def get_probabilities(self, text):
        try:
            detector = self.factory.create()
            if self.max_length:
                text = text[:self.max_length]
            detector.append(text)
            probs =  detector.get_probabilities()
            return {x.lang: x.prob for x in probs}
        except Exception as e:
            return {}
