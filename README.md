# *TrufflePig*
### A Steemit Curation Bot based on Natural Language Processing and Machine Learning

![test](https://travis-ci.org/SmokinCaterpillar/TrufflePig.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/SmokinCaterpillar/TrufflePig/badge.svg?branch=master)](https://coveralls.io/github/SmokinCaterpillar/TrufflePig?branch=master)

[Steemit](https://steemit.com) can be a tough place for minnows, as new users are often called. I had to learn this myself. Due to the incredible number of new posts published every minute, it is exceptionally difficult to stand apart from the crowd. Nice, well-researched, and well-crafted posts from minnows are often overlooked. Minnows do not benefit from influential followers to upvote their high-quality posts. Their contributions are lost long before any whale may notice them and turn these posts into trending topics.

User-based curation does have merrit and it is possible that posts receive the traction and recognition they deserve. I believe there is a way to support the Steemit content curators. A way in which high-quality content no longer goes unnoticed. I have developed a curation bot called `TrufflePig` to do exactly this using Natural Language Processing and Machine Learning. The deployed bot can be found here: https://steemit.com/@trufflepig

### The Concept

The idea is to use well-received posts as training examples to teach a Machine Learning Regressor (MLR) what high-quality Steemit content looks like. Once trained, the Machine Learning Regressor is used to identify high-quality posts which were missed by the curation community. These posts which receive less payment than they deserved are dubbed *truffles*.

The general idea of the system is as follows:

1. I train a Machine Learning regressor (MLR) using Steemit posts as inputs and the corresponding Steem Dollar (SBD) reward and the number of votes as outputs.

2. The MLR learns to predict potential payouts for new Steemit posts.

3. I compare the predicted payout with the actual payout of these recent Steemit posts (between 2 and 26 hours old). When the Machine Learning model predicts a high reward, where such a reward was not actually assigned to the post, I classify this post as an overlooked truffle.

### The Implementation

The Machine Learning Regression model is trained on posts older than 7 days which have already been paid. Features include spelling errors, post length, and readability scores. A post's content is modelled as a [Latent Semantic Indexing](https://de.wikipedia.org/wiki/Latent_Semantic_Analysis) projection. The final regressor is a multi-output [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

The bot uses the official [Steem Python](https://github.com/steemit/steem-python) library to scrape data from the steemit blockchain and to post a toplist of the daily found truffles using its trained model.

The bot works as follows: 

1. Older data is scraped from the blockchain (see `bchain.getdata.py`) or loaded from disk if possible. 

2. The scraped posts are filtered and preprocessed (see `preprocessing.py`). 

3. A model is trained on the processed data if one does not yet exist (see `model.py`) or is otherwise loaded from disk.

4. More recent data is scraped and checked for truffles using this trained model.

5. The bot publishes a toplist of truffles on which it both upvotes and comments (see `bchain.postdata.py`).

### Installation and Execution

Clone the project directory:
> `$ git clone https://github.com/SmokinCaterpillar/TrufflePig.git`

Add the project directory to your `PYTHONPATH`:
> `$ echo '$PYTHONPATH=$PYTHONPATH:<project-directory-path>' >> ~/.bash_profile & source ~/.bash_profile`

Start the bot using the provided `main.py` driver:
> `python main.py`

You can manually set the time the bot considers as `now` via  the --now configuration flag.
> `--now='2018-01-01-11:42:42'`.

By default, the bot will not post to the blockchain. To enable posting use the `--broadcast` flag. 

The bot's account information requires you to populate environment variables `STEEM_ACCOUNT`, `STEEM_POSTING_KEY`, and `STEEM_PASSWORD`.
`STEEM_PASSWORD` is optional and used only to encrypt the wallet file. The password *should not* be your Steemit masterpassword.

### Open Source Usage

The bot is open source and can be freely used for **non-commercial** (!) purposes. Please check the LICENSE file.

![trufflepig](https://raw.githubusercontent.com/SmokinCaterpillar/TrufflePig/master/img/trufflepig17_small.png)

*`TrufflePig`*

(The bot's avatar has been created using https://robohash.org/)
