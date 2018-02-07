import trufflepig.textfilters as tf


def test_filter_html_tags():
    result = tf.filter_html_tags('<jkdjdksakd>hi</img>')
    assert result == 'hi'



