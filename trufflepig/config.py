import os

NODE_URL = os.environ.get('STEEM_NODE_URL', 'https://api.steemit.com')

ACCOUNT = os.environ.get('STEEM_ACCOUNT', 'trufflepig')
PASSWORD = os.environ.get('STEEM_PASSWORD', os.environ.get('UNLOCK', None))
POSTING_KEY = os.environ.get('STEEM_POSTING_KEY', None)

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))