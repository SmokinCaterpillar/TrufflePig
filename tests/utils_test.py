import pytest

from steembase.exceptions import RPCError

from trufflepig.utils import progressbar, error_retry, none_retry


def test_progressbar():
    result = []
    for irun in range(100):
        result.append(progressbar(irun, 100, percentage_step=1))
    assert all(result)


def f():
    raise RPCError


def g():
    return None


def test_rpc_retry():
    with pytest.raises(RPCError):
        error_retry(f, sleep_time=0.01, errors=RPCError)()


def test_no_logrpc_retry():
    with pytest.raises(RPCError):
        error_retry(f, sleep_time=0.01, errors=RPCError,
                    not_log_errors=(RPCError,))()


def test_none_retry():
    result = none_retry(g, sleep_time=0.01)()
    assert result is None
