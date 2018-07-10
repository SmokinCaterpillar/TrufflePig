import time
import logging

from steem.account import Account
from trufflepig.utils import error_retry

logger = logging.getLogger(__name__)


WAIT = 24.42

THRESHOLD = 'threshold'


class Poster(object):
    """A class to allow for posting and taking care of posting intervals"""
    def __init__(self, steem, account, self_vote_limit=94, waiting_time=WAIT,
                 no_posting_key_mode=False):
        self.no_posting_key_mode = no_posting_key_mode
        self.waiting_time = waiting_time
        self.last_post_time = time.time() - self.waiting_time
        self.steem = steem
        self.account = account
        self.self_vote_limit=self_vote_limit
        logger.info('Poster ready for account {}, waiting time {} '
                    'and limit {}.'.format(account, waiting_time, self_vote_limit))

    def check_if_self_vote(self):
        voting_power = self.get_voting_power()
        return voting_power >= self.self_vote_limit

    def get_voting_power(self):
        acc = Account(self.account, self.steem)
        return acc.voting_power()

    def post(self, body, title, permalink, tags, self_vote=False):
        if self_vote == THRESHOLD:
            self_vote = self.check_if_self_vote()
        self.time2last_post()
        logger.info('Posting: `{}` (`{}`)\n{}'.format(title, permalink, body))
        if self.no_posting_key_mode:
            logger.warning('Test mode NOT TRYING TO POST')
        else:
            return error_retry(self.steem.commit.post, retries=10,
                        sleep_time=4, errors=Exception)(author=self.account,
                                               title=title,
                                               body=body,
                                               permlink=permalink,
                                               self_vote=self_vote,
                                               tags=tags)

    def time2last_post(self):
        now = time.time()
        diff = now - self.last_post_time
        if  diff < self.waiting_time:
            time.sleep(self.waiting_time - diff)
        self.last_post_time = time.time()

    def vote(self, author, permalink, weight):
        identifier = '@{author}/{permalink}'.format(author=author,
                                                    permalink=permalink)
        logger.info('Voting on {} with weight {}.'.format(identifier,weight))
        if self.no_posting_key_mode:
            logger.warning('Test mode NOT TRYING TO VOTE!')
        else:
            error_retry(self.steem.commit.vote, retries=5,
                        sleep_time=4)(identifier=identifier,
                                                weight=weight,
                                                account=self.account)

    def reply(self, body, parent_author, parent_permalink, self_vote=False,
              parent_vote_weight=0):
        identifier = '@{author}/{permalink}'.format(author=parent_author,
                                                    permalink=parent_permalink)

        if parent_vote_weight:
            self.vote(parent_author, parent_permalink, parent_vote_weight)

        if self_vote == THRESHOLD:
            self_vote = self.check_if_self_vote()

        self.time2last_post()
        logger.info('Replying to {} with\n{}'.format(identifier, body))
        if self.no_posting_key_mode:
            logger.warning('Test mode NOT TRYING TO REPLY')
        else:
            return error_retry(self.steem.commit.post, retries=10,
                        sleep_time=4, errors=Exception)("",
                                              body,
                                              json_metadata=None,
                                              author=self.account,
                                              reply_identifier=identifier,
                                              self_vote=self_vote)