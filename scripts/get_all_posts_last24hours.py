import logging
import pandas as pd
from steem import Steem

from trufflepig import config

import trufflepig.bchain.getdata as tpbg

logging.basicConfig(level=logging.INFO)

steem = Steem(nodes=config.NODES)

now = pd.datetime.utcnow()
end = now
start = end - pd.Timedelta(hours=24)
posts = tpbg.get_all_posts_between(start, end, steem)