import pytest
from steem.blockchain import Blockchain

from trufflepig import config
from trufflepig.bchain import getdata as tpbg
from trufflepig.bchain.mpsteem import MPSteem


@pytest.fixture(scope="session")
def steem():
    return MPSteem(nodes=config.NODES, no_broadcast=True)


@pytest.fixture
def bchain(steem):
    return Blockchain(steem)


@pytest.fixture
def temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test', numbered=True)
