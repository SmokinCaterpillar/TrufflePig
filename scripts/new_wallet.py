from steem import Steem
from trufflepig import config

def main():
    kwargs = dict(nodes=[config.NODE_URL],
                no_broadcast=True)

    steem = Steem(**kwargs)

    # if not steem.wallet.created():
    #     steem.wallet.newWallet()

    print('Unocking')
    wallet = steem.wallet
    wallet.unlock(pwd=config.PASSWORD)
    print('Adding Posting Key')
    try:
        wallet.addPrivateKey(config.POSTING_KEY)
    except ValueError:
        print('Key already present')
    print('Done')


if __name__ == '__main__':
    main()