# *TrufflePig*
### A Steemit Curation Bot based on Natural Language Processing and  Machine Learning

![test](https://travis-ci.org/SmokinCaterpillar/TrufflePig.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/SmokinCaterpillar/TrufflePig/badge.svg?branch=master)](https://coveralls.io/github/SmokinCaterpillar/TrufflePig?branch=master)

[Steemit](https://steemit.com) can be a tough place for minnows, as new users are often called. I had to learn this myself. Due to the incredible amount of new posts that are published by the minute, it is incredibly hard to stand out from the crowd. Often even nice, well-researched, and well-crafted posts of minnows get buried in the noise because they do not benefit from a lot of influential followers that could upvote their quality posts. Hence, their contributions are getting lost long before one or the other whale could notice them and turn them into trending topics.

However, this user based curation also has its merits, of course. You can become fortunate and your nice posts get traction and the recognition they deserve. Maybe there is a way to support the Steemit content curators such that high quality content does not go unnoticed anymore. In fact, I developed a curation bot called `TrufflePig` to do exactly this with the help of Natural Language Processing and Machine Learning. The deployed bot can be found here: https://steemit.com/@trufflepig

### The Concept

The basic idea is to use well paid posts of the past as training examples to teach a Machine Learning Regressor (MLR) how high quality Steemit content looks like. In turn, the trained MLR can be used to identify posts of high quality that were missed by the curation community and did receive much less payment than they deserved. We call this posts *truffles*.

The general idea of this bot is the following:

1. We train a Machine Learning regressor (MLR) using Steemit posts as inputs and the corresponding Steem Dollar (SBD) rewards and votes as outputs.

2. Accordingly, the MLR should learn to predict potential payouts for new, beforehand unseen Steemit posts.

3. Next, we can compare the predicted payout with the actual payouts of recent Steemit posts (between 2 and 26 hours old). If the Machine Learning model predicts a huge reward, but the post was merely paid at all, we classify this contribution as an overlooked truffle.

### The Implementation

The bot is trained on posts that are older than 7 days and, therefore, have already been paid. Features include style measures such as spelling errors, number of words, readability scores. Moreover, a post's content is modelled as a [Latent Semantic Indexing](https://de.wikipedia.org/wiki/Latent_Semantic_Analysis) projection. The final regressor is simply a multi-output [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

To scrape data from the steemit blockchain and to post a toplist of the daily found truffles the bot uses the official [Steem Python](https://github.com/steemit/steem-python) library.

The bot works as follows: First older data is scraped from the blockchain (see `bchain.getdata.py`) or, if possible, loaded from disk. Next, the scraped posts are filtered and preprocessed (see `preprocessing.py`). Subsequently, a model is trained on the processed data (see `model.py`) or, if possible, loaded from disk. Next, more recent data is scraped and checked for truffles. Finally, the bot publishes a toplist, upvotes, and comments on the truffles (see `bchain.postdata.py`).

### Installation and Execution

Simply, `git clone https://github.com/SmokinCaterpillar/TrufflePig.git` and add the project directory to your `PYTHONPATH`. The bot can be started with `python main.py`.

You can manually set the time the bot considers as `now` via `--now='2018-01-01-11:42:42'`. By default, the bot won't post to the blockchain. To enable posting use `--broadcast`. Moreover, the bot's account information needs to be set via environment variables: `STEEM_ACCOUNT`, `STEEM_POSTING_KEY`, and `STEEM_PASSWORD`. The latter is up to your choice and is only used to encrypt the wallet file. The password does not need to (and should not) be your Steemit masterpassword.

### Open Source Usage

The bot is open source and can be freely used for **non-commercial** (!) purposes. Please, check the LICENSE file.

![trufflepig](https://raw.githubusercontent.com/SmokinCaterpillar/TrufflePig/master/img/trufflepig17_small.png)

*`TrufflePig`*

(The bot's avatar has been created using https://robohash.org/)
