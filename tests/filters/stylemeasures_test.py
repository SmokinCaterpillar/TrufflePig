import trufflepig.filters.stylemeasures as tpsm


def test_count_paragraphs():
    result = tpsm.count_paragraphs('Hello \n\n World \n\n\n !')
    assert result == 3


def test_detect_language():
    detector = tpsm.LanguageDetector()
    result = detector.detect_language('die katze ist klein der hund auch')
    assert result == 'de'


def test_language_probs():
    detector = tpsm.LanguageDetector()
    result = detector.get_probabilities('die katze ist klein and I love mike and peter')
    assert len(result) > 1


def test_split_into_sentences():
    result = tpsm.split_into_sentences('Hi my name is! Slim Shady! Really? Yeah.')
    assert result == ['Hi my name is!', 'Slim Shady!', 'Really?', 'Yeah.']


def test_compute_average_sentence_length():
    result = tpsm.compute_average_sentence_length(['huhuh.', 'sbbbasdsads'])
    assert result == 8.5


def test_compute_sentence_length_variance():
    result = tpsm.compute_sentence_length_variance(['huhuh.', 'sbbbasdsads', 'jj djdjd', '1'])
    assert result == 13.25


def test_compute_average_punctiation():
    result = tpsm.compute_average_puncitation(['..jj', '..asdsad!!asdsd'])
    assert result == 3


def test_count_mistakes():
    counter = tpsm.SpellErrorCounter()
    result = counter.count_mistakes('hi hiw are you blockchain crypto cryptocurrency '
                                    'cryptocurrencies steem steemit bitcoin ethereum')
    assert result == 1


def test_count_connectors():
    result = tpsm.count_connectors(['indeed', 'i', 'like', 'hence'])
    assert result == 2


def test_count_pronouns():
    result = tpsm.count_pronouns(['me', 'and', 'myself', 'like',
                                  'to', 'go', 'to', 'her'])
    assert result == 2


def test_count_letters():
    result = tpsm.count_letters('hallo54 my namE!!')
    assert result == 11