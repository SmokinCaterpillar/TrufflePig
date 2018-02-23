# *TrufflePig*
### A Steemit Curation Bot based on Natural Language Processing and  Machine Learning

![test](https://travis-ci.org/SmokinCaterpillar/TrufflePig.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/SmokinCaterpillar/TrufflePig/badge.svg?branch=master)](https://coveralls.io/github/SmokinCaterpillar/TrufflePig?branch=master)

This is a steemit curation bot based on Natural Language Processing and Machine Learning.
The deployed bot can be found here: https://steemit.com/@trufflepig

The basic idea is to use well paid posts of the past as training examples to teach a Machine Learning Regressor (MLR) how high quality Steemit content looks like. In turn, the trained MLR can be used to identify posts of high quality that were missed by the curation community and did receive much less payment than they deserved. We call this posts *truffles*.

The general idea of this bot is the following:

1. We train a Machine Learning regressor (MLR) using Steemit posts as inputs and the corresponding Steem Dollar (SBD) rewards and votes as outputs.

2. Accordingly, the MLR should learn to predict potential payouts for new, beforehand unseen Steemit posts.

3. Next, we can compare the predicted payout with the actual payouts of recent Steemit posts (between 24 and 48 hours old). If the Machine Learning model predicts a huge reward, but the post was merely paid at all, we classify this contribution as an overlooked truffle.

The bot is trained on posts that are older than 7 days and, therefore, have already been paid. Features include style measures such as spelling errors, number of words, readability scores. Moreover, a post's content is modelled as a [Latent Semantic Indexing](https://de.wikipedia.org/wiki/Latent_Semantic_Analysis) projection. The final regressor is simply a multi-output [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

To scrape data from the steemit blockchain and to post a toplist of the daily found truffles the bot uses the official [Steem Python](https://github.com/steemit/steem-python) library.

The bot works as follows: First older data is scraped from the blockchain (see the `bchain.getdata.py`) or, if possible, loaded from disk. Next a model is trained (see `model.py`) or, if possible, loaded from disk. Next, more recent data is scraped and checked for truffles. Finally, the bot publishes a toplist, upvotes, and comments on the truffles (see `bchain.postdata.py`).

The bot is open source and can be freely used for **non-commercial** (!) purposes. Please, check the LICENSE file.


