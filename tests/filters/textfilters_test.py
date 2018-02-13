import trufflepig.filters.textfilters as tptf


def test_filter_html_tags():
    result = tptf.filter_html_tags('<jkdjdksakd>hi</img>')
    assert result == 'hi'


def test_filter_images_and_links():
    result = tptf.filter_images_and_links('Lookat ![j kjds](wehwjrkjewrk.de), yes [iii](jlkajddjsla), and '
                        '![images (17).jpg](https://steemitimages.com/DQmQF5BxHtPdPu1yKipV67GpnRdzemPpEFCqB59kVXC6Ahy/images%20(17).jpg)')
    assert result == 'Lookat , yes , and '


def test_filter_urls():
    result = tptf.filter_urls('I like www.pipes.com')
    assert result == 'I like '


def test_filter_special_characters():
    result = tptf.filter_special_characters('Hi//)(&(/%( \n\n\t)))""""""!!!.')
    assert result == 'Hi \n\n\t!!!.'


def test_filter_formatting():
    result = tptf.filter_formatting('Hi&nbsphey aligncenter nbsp styletextalign kk')
    assert result == 'Hi hey     kk'


def test_replace_newlines():
    result = tptf.replace_newlines('Hi \n\n\tee')
    assert result == 'Hi ee'


def test_filter_punctuation():
    result = tptf.filter_punctuation('hi. my. name. is yolo!;;k;')
    assert result == 'hi my name is yolok'

