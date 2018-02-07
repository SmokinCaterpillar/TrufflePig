import trufflepig.stylemeasures as tpsm


def test_count_paragraphs():
    result = tpsm.count_paragraphs('Hello \n\n World \n\n\n !')
    assert result == 3


def test_detect_languade():
    result = tpsm.detect_language('die katze ist klein der hund auch')
    assert result == 'de'