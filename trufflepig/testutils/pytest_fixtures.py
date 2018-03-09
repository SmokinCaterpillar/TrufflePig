import pytest
from steem.blockchain import Blockchain

from trufflepig import config
from trufflepig.bchain import getdata as tpbg


@pytest.fixture()
def steem_kwargs():
    return dict(nodes=config.NODES,
                no_broadcast=True)


@pytest.fixture
def steem(steem_kwargs):
    return tpbg.Steem(**steem_kwargs)


@pytest.fixture
def bchain(steem):
    return Blockchain(steem)


@pytest.fixture
def temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test', numbered=True)
