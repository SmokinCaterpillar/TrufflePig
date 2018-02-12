import numpy as np
import pandas as pd

AUTHORS = ['mary', 'mike', 'maureen', 'michael', 'manuel', 'miriam']

WORDS = """
Being a minnow on Steemit can be tough, your well-crafted posts may get unnoticed and buried in the noise. Accordingly, I made a [proposition to develop a new Artificial Intelligence](https://steemit.com/steemit/@smcaterpillar/proposing-trufflepig-an-artificial-intelligence-to-help-minnows-and-content-curators) (AI), codename `TrufflePig,` to help the Steemit curation community by identifying overlooked and undervalued content. The best part of this proposition? I already have some data to show! I already experimented a bit and developed a prototype. Here I want to briefly discuss and summarize my initial results. **So feel free to upvote, resteem, and comment on this post to support the further development of the bot.**

The general idea of this Artificial Intelligence is the following:

1. We train a Machine Learning regressor (MLR) using Steemit posts as inputs and the corresponding Steem Dollar (SBD) rewards as outputs.

2. Accordingly, the MLR should learn to predict potential payouts for new, beforehand unseen Steemit posts.

3. Next, we can compare the predicted payout with the actual payouts of the Steemit posts. If the Machine Learning model predicts a huge reward, but the post was merely paid at all, we classify this contribution as an overlooked truffle or hidden gem.

As we will see later in the examples, this works surprisingly well for a first prototype!
In an online and live version of the bot we would publish hidden gems within the 7 days payout period (maybe posts between 2 and 6 days old) such that the community still has some time to give the truffles a proper reward.

![robot](https://raw.githubusercontent.com/SmokinCaterpillar/blog/master/2018_02_04_truffle_pig/robot.png)

*Image from [goodfreephotos](https://www.goodfreephotos.com/vector-images/blue-robot-vector-art.png.php)*


## Data Scraping and Pre-Processing

I used the [official steem Python library](http://steem.readthedocs.io/en/latest/) to download about 38,000 quite recent posts from the Steemit blockchain. Next, I performed some pre-processing and filtering. For example, I filtered images and links using regular expressions like this one:

```
def filter_images_and_links(text):
    return re.sub('!?\[[-a-zA-Z0-9?@: %._\+~#=/()]*\]\([-a-zA-Z0-9?@:%._\+~#=/()]+\)', '', text)
```

Next, I only kept posts with at least 1000 characters left. Our AI should learn to identify high quality written content. Meme and video posts that go viral are not considered and ignored by the filtering. Moreover, I am only interested in contributions written in English. To filter for English only posts, I used the [language detect library](https://pypi.python.org/pypi/langdetect?).

## Feature Engineering

Usually the most difficult and involved part of engineering a Machine Learning application is the proper design of features. How are we going to represent the Steemit posts so they can be understood by an Artificial Intelligence?

It is important that we use features that represent the content and quality of the post. We do not want to use author specific features such as the number of followers or past author payouts. Although I am quite certain (and I'll test this sometime soon) that these are incredibly predictive features of future payouts, these do not help us to identify overlooked and buried truffles.

I used some features that encode the style of the posts, such as number of paragraphs, average words per sentence, or spelling mistakes. Clearly, posts with many spelling errors are usually not high-quality content and are, to my mind, a pain to read. Orthography checks were performed using [PyEnchant](http://pythonhosted.org/pyenchant/).

Still, the question remains, how are we going to encode the content of the post? How to represent the topic someone chose and the story an author told? The most simple encoding that is quite often used is the so called ['term frequency inverse document frequency'](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (tf-idf). This technique basically encodes each document, so in our case Steemit posts, by the particular words that are present and weighs them by their (heuristically) normalized frequency of occurrence. However, this encoding produces vectors of enormous length with one entry for each unique word in all documents. Hence, most entries in these vectors are zeroes anyway because each document contains only a small subset of all potential words. For instance, if there are 150,000 different unique words in all our Steemit posts, each post will be represented by a vector of length 150,000 with almost all entries set to zero. Even if we filter and ignore very common words such as `the` or `a` we could easily end up with vectors having 30,000 or more dimensions.

Such high dimensional input is usually not very useful for Machine Learning. We rather want a much lower dimensionality than the number of training documents to effectively cover our data space. Accordingly, we need to reduce the dimensionality of our Steemit post representation. A widely used method is [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA), often also called Latent Semantic Indexing (LSI). Thanks to the [gensim project](https://radimrehurek.com/gensim/) there exists a Python version of this algorithm. LSI compression of the feature space is achieved by applying a Singular Value Decomposition (SVD) on top of the previously described word frequency encoding. A nice side effect of LSI is that the compressed features can be interpreted as topics and we can identify which words are most important (contributing positively or negatively) to form such a topic.

For example, in our (training) dataset the first identified topic by the gensim LSI is best represented by the words `post`, `latest`, `sp` (I presume short for steem power) and `flag`. Most likely this topic is about curating and voting of Steemit posts. We all know that talking about Steemit itself is a quite popular topic among Steemians. Another topic is represented by the words `bitcoin`, `yang` and `cryptocurrency`. Makes sense, doesn't it? The first 15 found topics are given in the image below.

![topics](https://raw.githubusercontent.com/SmokinCaterpillar/blog/master/2018_02_04_truffle_pig/topics.png)

In summary, the final Steemit post representation consist of the LSI topic dimensionality reduction (I chose an arbitrary number of 100 dimensions) in combination with the previously mentioned style features, such as number of spelling errors.

## Training of the Truffle Pig AI

Next, I randomly divided the data set into training (80%) and testing samples (20%). We use the training input vectors (topic reduction + style features) and the corresponding post payouts in SBD to train a Machine Learning regressor. I chose a [Random Forest regressor from scikit learn](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) because these are usually very robust to noise and do not require excessive tweaking of hyper parameters (in fact I did none!).

Our working hypothesis here is that the Steemit community can be trusted with their judgment. So whatever post was given a high payout is assumed to be high quality content -- and crap doesn't really make it to the top. Well, I know that there are some whale wars going on and there may be some exceptions to this rule, but we just treat those cases as noise in our dataset. Yet, we also assume that the Steemit community may miss some high quality posts from time to time. So there are potentially good posts out there that were not rewarded enough.

During training the Machine Learning model was able to pick up some rules and good mathematical representations of the training set as shown in the figure below and given by the model's performance score. The score is measured as `1-u/v`, where `u` is the residual sum of squares (`sum [(y_true - y_pred)^2]`) and `v` is the total sum of squares (`sum [(y_true - mean[y_true]))^2]`). The score on the training set is quite high with 0.9. If we predicted perfectly, the score would be 1.0. By the way, to get a feeling for this value, a score of 0 would mean we are as good -or rather as bad- as the predictor that always assumes the average payout. Of course, the score could in theory become negative because we could be arbitrarily awful in predicting payouts; for example, predicting 1 Mio. SBD for each post would be pretty stupid.

![training](https://raw.githubusercontent.com/SmokinCaterpillar/blog/master/2018_02_04_truffle_pig/training.png)

Moreover, you can look at the true reward in SDB (x-axis) versus the predicted one (y-axis) in the figure above. The identity line is shown in red. As you can see, the training predictions cluster nicely around this line. However, we do find some support for our working hypothesis: High rewards mean good posts in general, but writing a good posts does not automatically yield a high reward because the Steemit curation crowd might have missed it. How so? First, as shown in the figure for higher true rewards, the model usually underestimates the payout in SBD, and, secondly, the regressor predicted some high rewards for posts that almost did not get any reward at all.

Let us now use our trained truffle pig on the previously unknown test data. As expected, the performance is worse than for the training set, but still quite ok with a score value of 0.23. The data cloud no longer clusters so nicely along the identity line as for the training set, look at the figure below. Yet, in our model's defense, these are new posts that it did not see before and all Steemians know curation is hard!

![test](https://raw.githubusercontent.com/SmokinCaterpillar/blog/master/2018_02_04_truffle_pig/test.png)

## Do we find some hidden Truffles?

Let us come to the pressing question at hand: Did we find some hidden gems in the test set? Are there high quality posts that did not receive the reward they deserved? I would say, yes, for a first attempt the results are surprisingly convincing. Our Machine Learning pig did dig up some truffles. But, judge for yourself.

For example, among the top overlooked posts is the following one, which was given about 5 SDB, but our algorithm thought it is worth much more, that is 64 SBD. Surprisingly, its topic is very much related to the purpose of `TrufflePig`. It is titled  ["Re-Thinking Curation"](https://steemit.com/curation/@weaselhouse/re-thinking-curation) and discusses the problems of the current curation system. The author makes suggestion on how to increase visibility of high quality content by changes to the voting structure:

> "Ideally, only four numbers should matter when it comes to content curation (and none of those numbers are the SP of the person casting the vote). In fact, the four numbers that matter the most are right in front of us on every post:
> How many times has a post been seen?
> How many upvotes has it gained?
> How many downvotes has it gained?
> How many comments does it have?"

Other truffle examples are ["What I found in the Last Jedi"](https://steemit.com/movies/@talanhorne/what-i-found-in-the-last-jedi), that by the time of scraping was given less than 5 SBD, but according to the `TrufflePig` deserves about 58 SBD, or the short story series ["The Shanghai Songbird"](https://steemit.com/steempulp/@cheah/the-shanghai-songbird-part-2) with about 8 SBD paid versus 40 Dollars predicted. Nevertheless, the pig did dig up some controversial matters such as this post with the title ["In Defense of Aziz Ansari"](https://steemit.com/metoo/@trending/in-defense-of-aziz-ansari) (reward 1 SBD versus 87 predicted) or a very explicit story about a pornstar named Faye Reagan, I spare you the link here because it's very NSFW :-D

This concludes my short showcasing of the `TrufflePig` AI prototype. Still, there is a lot of room for improvement like better feature representations or trying out other Machine Learning methods. Moreover, I need to turn this offline model into an online bot that is regularly trained and regularly votes on and reports truffle posts. In fact, I already started the development in [my Github repository](https://github.com/SmokinCaterpillar/TrufflePig). Hopefully, the `TrufflePig` will support the Steemit curation crowd soon and help to give proper rewards to quality content that deserves it.

In the meantime **you can upvote, resteem, and comment on this and [my proposition post](https://steemit.com/steemit/@smcaterpillar/proposing-trufflepig-an-artificial-intelligence-to-help-minnows-and-content-curators)**! Please, do not let this one become an overlooked truffle... although I do like recursion, but I digress. So spread the word, and vote for some initial funding for the codebase and server setup! Of course, I will keep blogging about the bot's further development, so stay tuned for a detailed roadmap coming soon!

"""

WORDS = WORDS.split(' ')

TAGS = ['hello', 'world', 'computers', 'crypto',
        'maybecrypto', 'anothercrypto', 'moooorecrypto']


def create_post():
    nwords = np.random.randint(3, 1000)
    randwords = np.random.randint(0, len(WORDS), nwords)
    body = ' '.join(WORDS[x] for x in randwords)

    nwords = np.random.randint(1, 12)
    randtitle = np.random.randint(0, len(WORDS), nwords)
    title = ' '.join(WORDS[x] for x in randtitle)
    permalink = title.replace(' ', '-').lower()

    reward = np.random.rand() * 100
    votes = np.random.randint(0, 500)

    created = pd.datetime.utcnow()

    randtags = np.random.randint(0, len(TAGS), 5)
    tags = tuple(set(TAGS[x] for x in randtags))

    randauthor = np.random.randint(0, len(AUTHORS))
    author = AUTHORS[randauthor]

    post = {
            'title': title,
            'reward': reward,
            'votes': votes,
            'created': created,
            'tags': tags,
            'body': body,
            'author': author,
            'permalink': permalink,
        }

    return post


def create_n_random_posts(n, seed=42):
    np.random.seed(seed)
    return [create_post() for _ in range(n)]