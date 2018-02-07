import re

def filter_html_tags(text):
    return re.sub('</?[a-z]{1,11}>', '', text)

