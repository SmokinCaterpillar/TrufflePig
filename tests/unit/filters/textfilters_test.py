import trufflepig.filters.textfilters as tptf


def test_filter_html_tags():
    result = tptf.filter_html_tags('<jkdjdksakd>hi</img>')
    assert result == 'hi'


def test_filter_images_and_links():
    result = tptf.filter_images_and_links('Lookat ![j kjds](wehwjrkjewrk.de), yes [iii](jlkajddjsla), and '
                        '![images (17).jpg](https://steemitimages.com/DQmQF5BxHtPdPu1yKipV67GpnRdzemPpEFCqB59kVXC6Ahy/images%20(17).jpg)')
    assert result == 'Lookat , yes iii, and '


def test_get_image_urls():
    result = tptf.get_image_urls('Lookat ![j kjds](wehwjrkjewrk.de), yes [iii](jlkajddjsla), and '
                        '<img   src="hellokitty.com/hello">'
                        '![images (17).jpg](https://steemitimages.com/DQmQF5BxHtPdPu1yKipV67GpnRdzemPpEFCqB59kVXC6Ahy/images%20(17).jpg)')
    assert result == ['wehwjrkjewrk.de', 'hellokitty.com/hello',
                      'https://steemitimages.com/DQmQF5BxHtPdPu1yKipV67GpnRdzemPpEFCqB59kVXC6Ahy/images%20(17).jpg']


def test_filter_urls():
    result = tptf.filter_urls('I like www.pipes.com')
    assert result == 'I like '


def test_filter_special_characters():
    result = tptf.filter_special_characters('Hi//)(&(/%( \n\n\t)))""""""!!!.')
    assert result == 'Hi \n\n\t!!!.'


def test_filter_formatting():
    result = tptf.filter_formatting('Hi&nbsphey aligncenter nbsp Styletextalign kkhspace10')
    assert result == 'Hihey    kk'


def test_replace_newlines():
    result = tptf.replace_newlines('Hi \n\n\tee')
    assert result == 'Hi ee'


def test_filter_punctuation():
    result = tptf.filter_punctuation('hi. my. name. is yolo!;;k;')
    assert result == 'hi my name is yolok'


def test_is_in_tags():
    result = tptf.is_in_filter_tags(['hi', 'ho'], {'ha', 'hi'})
    assert result


def test_is_in_tags_typerror():
    result = tptf.is_in_filter_tags(['hi', ['ho']], {'ha', 'hi'})
    assert result


def test_voted_by():
    assert tptf.voted_by([{'voter': 'cheetah'}], {'cheetah'})


def test_filter_headdings():
    text= """# heading nheadings
heyho

#### heading123213213232

#################### dksajds
    
jdd
<h4> lkjsdsak!"§$$ </h5>

<H5> jjjjjj </H5>
"""
    new_text = tptf.filter_headings(text)
    expected = """heyho

#################### dksajds
    
jdd"""
    assert new_text == expected


def test_filter_quotes():
    text= """I like this
    
      > lksajdlksajdls743289473()/)(   /((/dads"!§"!§  432     )(/)(
      4 > 3
      
      Heyho!
      >When the total post value reaches $75 the winner will be chosen by random draw on @topkpop's Friday evening radio show. We will announce the winner and I will make contact through DM to find out where to send your awesome coin! Thank you all for your understanding. I want to support the Mothership and also not lose my shirt on my coin lol!!

>Everyone is welcome to support myself and themothership project But in order to be eligible to win the coin you must:

>Follow Me
      """
    new_text = tptf.filter_quotes(text)

    expected = """I like this
    
      4 > 3
      
      Heyho!


      """

    assert new_text == expected
