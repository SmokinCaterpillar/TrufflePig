import os

# The steemit nodes to load data from
NODE_URL = os.environ.get('STEEM_NODE_URL', 'https://api.steemit.com')
NODE_URL2 = os.environ.get('STEEM_NODE_URL2', 'https://steemd.privex.io')
NODE_URL3 = os.environ.get('STEEM_NODE_URL3', 'https://api.steem.house')
NODE_URL4 = os.environ.get('STEEM_NODE_URL4', 'steemd.minnowsupportproject.org')
NODE_URL5 = os.environ.get('STEEM_NODE_URL5', 'steemd.pevo.science')
NODE_URL6 = os.environ.get('STEEM_NODE_URL6', 'rpc.curiesteem.com')
NODE_URL7 = os.environ.get('STEEM_NODE_URL7', 'seed.bitcoiner.me')
NODES = [x for x in (NODE_URL, NODE_URL2, NODE_URL3,
                     NODE_URL4, NODE_URL5, NODE_URL6,
                     NODE_URL7) if x]

# The steemit bot account and password
ACCOUNT = os.environ.get('STEEM_ACCOUNT', 'trufflepig')
PASSWORD = os.environ.get('STEEM_PASSWORD', None)
if PASSWORD:
    # The wallet creation needs it in this very explicit way :-(
    os.environ['UNLOCK'] = PASSWORD
POSTING_KEY = os.environ.get('STEEM_POSTING_KEY', None)
ACTIVE_KEY = os.environ.get('STEEM_ACTIVE_KEY', None)

# Useful helper constant
PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))