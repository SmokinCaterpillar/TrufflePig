from trufflepig.utils import progressbar


def test_progressbar():
    result = []
    for irun in range(100):
        result.append(progressbar(irun, 100, percentage_step=1))
    assert all(result)
