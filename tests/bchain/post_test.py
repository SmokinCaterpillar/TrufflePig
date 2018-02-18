import pandas as pd

from tests.fixtures import random_data

import trufflepig.bchain.posts as tbpo
import trufflepig.preprocessing as tppp


def test_comment():
    post = random_data.create_n_random_posts(1)[0]

    result = tbpo.truffle_comment(reward=post['reward'],
                                  votes=post['votes'],
                                  rank=1,
                                  top10_link='www.example.com')

    assert result


def test_top10_post():
    posts = random_data.create_n_random_posts(10)
    df = pd.DataFrame(posts)
    df = tppp.preprocess(df, ncores=1)

    date = pd.datetime.utcnow().date()

    title, post = tbpo.top10_post(top10_authors=df.author,
                             top10_permalinks=df.permalink,
                             top10_titles=df.title,
                             top10_filtered_bodies=df.filtered_body,
                             top10_votes=df.votes,
                             top10_rewards=df.reward,
                             title_date=date)

    assert post
    assert title