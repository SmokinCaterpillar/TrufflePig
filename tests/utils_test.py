import pytest

from steembase.exceptions import RPCError

from trufflepig.utils import progressbar, rpcerror_retry


def test_progressbar():
    result = []
    for irun in range(100):
        result.append(progressbar(irun, 100, percentage_step=1))
    assert all(result)


def f():
    raise RPCError


def test_rpc_retry():
    with pytest.raises(RPCError):
        rpcerror_retry(f, sleep_time=0.01)()
