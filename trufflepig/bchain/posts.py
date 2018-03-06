import numpy as np

import trufflepig.filters.textfilters as tftf


TRUFFLE_LINK = 'https://steemit.com/steemit/@smcaterpillar/trufflepig-introducing-the-artificial-intelligence-for-content-curation-and-minnow-support'
TRUFFLE_IMAGE_SMALL = '![trufflepig](https://raw.githubusercontent.com/SmokinCaterpillar/TrufflePig/master/img/trufflepig17_small.png)'
TRUFFLE_IMAGE = '![trufflepig](https://raw.githubusercontent.com/SmokinCaterpillar/TrufflePig/master/img/trufflepig17.png)'
DELEGATION_LINK = 'https://v2.steemconnect.com/sign/delegateVestingShares?delegator=&delegatee=trufflepig&vesting_shares={shares}%20VESTS'
QUOTE_MAX_LENGTH = 496
TAGS = ['steemit', 'steem', 'minnowsupport', 'upvote', 'community']

BODY_PREFIX = ''  # to announce tests etc.


def truffle_comment(reward, votes, rank, topN_link, truffle_link=TRUFFLE_LINK, truffle_image_small=TRUFFLE_IMAGE_SMALL):
    """Creates a comment made under an upvoted toplist post"""
    post = """**Congratulations!** Your post has been selected as a daily Steemit truffle! It is listed on **rank {rank}** of all contributions awarded today. You can find the [TOP DAILY TRUFFLE PICKS HERE.]({topN_link}) 
    
I upvoted your contribution because to my mind your post is at least **{reward} SBD** worth and should receive **{votes} votes**. It's now up to the lovely Steemit community to make this come true.

I am `TrufflePig`, an Artificial Intelligence Bot that helps minnows and content curators using Machine Learning. If you are curious how I select content, [you can find an explanation here!]({truffle_link})
    
Have a nice day and sincerely yours,
{truffle_image_small}
*`TrufflePig`*
    """
    post = BODY_PREFIX + post

    return post.format(reward=int(reward), votes=int(votes), topN_link=topN_link,
                       truffle_link=truffle_link, rank=rank,
                       truffle_image_small=truffle_image_small)


def topN_list(topN_authors, topN_permalinks, topN_titles,
              topN_filtered_bodies, topN_image_urls,
              topN_rewards, topN_votes, quote_max_length, nstart=1):
    """Creates a toplist string"""
    topN_entry="""{rank}. [{title}](https://steemit.com/@{author}/{permalink})  --  **by @{author} with an estimated worth of {reward:d} SBD and {votes:d} votes**
    
    {image}{quote}

"""

    result_string = ""

    iterable = zip(topN_authors, topN_permalinks, topN_titles,
                   topN_filtered_bodies, topN_image_urls,
                   topN_rewards, topN_votes)

    for idx, (author, permalink, title, filtered_body, img_urls, reward, votes) in enumerate(iterable):
        rank = idx + nstart
        quote = '>' + filtered_body[:quote_max_length].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') + '...'
        title = tftf.replace_newlines(title)
        title = tftf.filter_special_characters(title)
        if len(img_urls) >= 1:
            imgstr = """ <div class="pull-right"><img src="{img}" /></div>\n\n    """.format(img=img_urls[0])
        else:
            imgstr=''
        entry = topN_entry.format(rank=rank, author=author, permalink=permalink,
                                   title=title, quote=quote, votes=int(votes),
                                   reward=int(reward), image=imgstr)
        result_string += entry
    return result_string


def simple_topN_list(topN_authors, topN_permalinks, topN_titles,
                     topN_rewards, topN_votes, nstart):
    """Creates a toplist for lower ranks"""
    topN_entry="""\n {rank}: [{title}](https://steemit.com/@{author}/{permalink}) (by @{author}, {reward:d} SBD, {votes:d} votes)\n"""

    result_string = ""

    iterable = zip(topN_authors, topN_permalinks, topN_titles,
                   topN_rewards, topN_votes)

    for idx, (author, permalink, title, reward, votes) in enumerate(iterable):
        rank = idx + nstart
        title = tftf.replace_newlines(title)
        title = tftf.filter_special_characters(title)
        entry = topN_entry.format(rank=rank, author=author, permalink=permalink,
                                   title=title, votes=int(votes),
                                   reward=int(reward))
        result_string += entry
    return result_string


def get_delegation_link(steem_per_mvests, steem_powers=(10, 50, 100, 500, 1000, 5000)):
    """Returns a dictionary of links to delegate SP"""
    link_dict = {}
    for steem_power in steem_powers:
        shares = np.round(steem_power / steem_per_mvests * 1e6, 3)
        link_dict['sp'+str(steem_power)] = DELEGATION_LINK.format(shares=shares)
    return link_dict


def topN_post(topN_authors, topN_permalinks, topN_titles,
              topN_filtered_bodies, topN_image_urls,
              topN_rewards, topN_votes, title_date,
              steem_per_mvests=490,
              truffle_link=TRUFFLE_LINK,
              truffle_image=TRUFFLE_IMAGE,
              quote_max_length=QUOTE_MAX_LENGTH):
    """Craetes the truffle pig daily toplist post"""
    title = """Today's Truffle Picks: Quality Steemit Posts that deserve more Rewards and Attention! ({date})"""

    post=""" ## Daily Truffle Picks
    
It's time for another round of truffles I found digging in the streams of this beautiful platform!

For those of you who do not know me: My name is *TrufflePig*. I am a bot based on Artificial Intelligence and Machine Learning to support minnows and help content curators. I was created and am being maintained by @smcaterpillar. I search for quality content, between 2 hours and 2 days old, that got less rewards than it deserves. I call these posts truffles, publish a daily top list, and upvote them. Now it is up to you to give these posts the attention they deserve. If you are curious how I select content, [you can find an explanation here.]({truffle_link})
    
Please, be aware that the list below has been automatically generated by a Machine Learning algorithm that was trained on payouts of previous contributions of the Steemit community. Of course, **this algorithm can make mistakes**. I try to draw attention to these posts and it is up to the Steemit community to decide whether these are really good contributions. Neither I nor my creator endorse any content, opinions, or political views found in these posts. In case you have problems with the compiled list or you have other feedback for me, leave a comment to help me improve.
    
# The Top 10 Truffles

Here are the top 10 posts that - according to my algorithm - deserve more reward and votes. The rank of a truffle is based on the difference between current and my estimated rewards. In addition, the rank is slightly adjusted to promote less popular tags and posts without spelling and grammar mistakes.

{topN_truffles}


### You didn't make it into the top list this time?

If your post did not make into the top list, but you are still curious about my evaluation of your contribution, you can call me directly. Just reply to your own post with @trufflepig. I will answer the call within the next 24 hours.

## You can Help and Contribute

By checking, upvoting, and resteeming the found truffles from above, you help minnows and promote good content on Steemit. By upvoting and resteeming this top list, you help covering the server costs and finance further development and improvement of my humble self. 

**NEW**: You may further show your support for me and all the found truffles by [**following my curation trail**](https://steemauto.com/dash.php?trail=trufflepig&i=1) on SteemAuto!

## Delegate and Invest in the Bot

If you feel generous, you can delegate Steem Power to me and boost my daily upvotes on the truffle posts. In return, I will provide you with a *small* compensation for your trust in me and your locked Steem Power. **Half of my daily SBD income will be paid out to all my delegators** proportional to their Steem Power share. Payouts will start 3 days after your delegation.

Click on one of the following links to delegate **[10]({sp10}), [50]({sp50}), [100]({sp100}), [500]({sp500}), [1000]({sp1000}),** or even **[5000 Steem Power]({sp5000})**. Thank You!

Cheers,

{truffle_image}

*`TrufflePig`*
    """
    link_dict = get_delegation_link(steem_per_mvests=steem_per_mvests)

    topN_truffles = topN_list(topN_authors=topN_authors,
                              topN_permalinks=topN_permalinks,
                              topN_titles=topN_titles,
                              topN_filtered_bodies=topN_filtered_bodies,
                              topN_image_urls=topN_image_urls,
                              topN_rewards=topN_rewards,
                              topN_votes=topN_votes,
                              quote_max_length=quote_max_length)

    title = title.format(date=title_date.strftime('%d.%m.%Y'))
    post = post.format(topN_truffles=topN_truffles,
                          truffle_image=truffle_image,
                          truffle_link=truffle_link,
                          **link_dict)
    post = BODY_PREFIX + post
    return title, post


def topN_comment(topN_authors, topN_permalinks, topN_titles,
                 topN_rewards, topN_votes, nstart=11):
    """Creates the toplist comment for lower ranks"""
    post = """If you cannot get enough truffles, here are ranks 11 till 25:
    
{topN_truffles}
    """

    topN_truffles = simple_topN_list(topN_authors=topN_authors,
                                     topN_permalinks=topN_permalinks,
                                     topN_titles=topN_titles,
                                     topN_votes=topN_votes,
                                     topN_rewards=topN_rewards,
                                     nstart=nstart)
    post = post.format(topN_truffles=topN_truffles)
    return post


def on_call_comment(author, reward, votes, topN_link, truffle_link=TRUFFLE_LINK, truffle_image_small=TRUFFLE_IMAGE_SMALL):
    """Creates a comment made under an upvoted toplist post"""
    post = """Thanks for calling @{author}! Here is a small upvote for this post and my opinion about it.
    
To my mind this post is at least **{reward} SBD** worth and should receive **{votes} votes**.

By the way, you can find [TODAY'S TRUFFLE PICKS HERE.]({topN_link}) 

I am `TrufflePig`, an Artificial Intelligence Bot that helps minnows and content curators using Machine Learning. If you are curious how I evaluate content, [you can find an explanation here!]({truffle_link})
    
Have a nice day and sincerely yours,
{truffle_image_small}
*`TrufflePig`*
    """
    post = BODY_PREFIX + post

    return post.format(author=author, reward=int(reward), votes=int(votes), topN_link=topN_link,
                       truffle_link=truffle_link,
                       truffle_image_small=truffle_image_small)
