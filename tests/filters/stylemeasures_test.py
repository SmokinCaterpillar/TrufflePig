import trufflepig.filters.stylemeasures as tpsm


def test_count_paragraphs():
    result = tpsm.count_paragraphs('Hello \n\n World \n\n\n !')
    assert result == 3


def test_detect_language():
    detector = tpsm.LanguageDetector()
    result = detector.detect_language('die katze ist klein der hund auch')
    assert result == 'de'


def test_split_into_sentences():
    result = tpsm.split_into_sentences('Hi my name is! Slim Shady! Really? Yeah.')
    assert result == ['Hi my name is!', 'Slim Shady!', 'Really?', 'Yeah.']


def test_compute_average_sentence_length():
    result = tpsm.compute_average_sentence_length(['huhuh.', 'sbbbasdsads'])
    assert result == 8.5


def test_compute_sentence_length_variance():
    result = tpsm.compute_sentence_length_variance(['huhuh.', 'sbbbasdsads', 'jj djdjd', '1'])
    assert result == 13.25


def test_count_mistakes():
    counter = tpsm.SpellErrorCounter()
    result = counter.count_mistakes('hi hiw are you')
    assert result == 1