from steem import Steem
from steem.commit import Commit
from steem.steemd import Steemd


class MPSteem(Steem):
    """Multiprocessing safe Steem"""
    def __init__(self, nodes: list, no_broadcast=False, **kwargs):
        self.nodes = nodes.copy()
        self.no_broadcast = no_broadcast
        self.kwargs = kwargs.copy()
        super().__init__(nodes=nodes, no_broadcast=no_broadcast, **kwargs)

    def reconnect(self):
        """Creates a new Steemd and Commit"""
        self.steemd = Steemd(
            nodes=self.nodes.copy(),
            **self.kwargs.copy()
        )
        self.commit = Commit(
            steemd_instance=self.steemd,
            no_broadcast=self.no_broadcast,
            **self.kwargs.copy()
        )

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['steemd']
        del result['commit']
        return result

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.reconnect()
