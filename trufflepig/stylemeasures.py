import langdetect


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
