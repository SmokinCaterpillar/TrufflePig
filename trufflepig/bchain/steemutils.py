from steem import Steem
from steem.commit import Commit
from steem.steemd import Steemd


class MPSteem(Steem):
    def __init__(self, nodes: list, no_broadcast=False, **kwargs):
        self.nodes = nodes.copy()
        self.no_broadcast = no_broadcast
        self.kwargs = kwargs.copy()
        super().__init__(nodes=nodes, no_broadcast=no_broadcast, **kwargs)

    def connect(self):
        self.steemd = Steemd(
            nodes=self.nodes,
            **self.kwargs
        )
        self.commit = Commit(
            steemd_instance=self.steemd,
            no_broadcast=self.no_broadcast,
            **self.kwargs
        )

    def disconnect(self):
        self.steemd = None
        self.commit = None

    def reconnect(self):
        self.disconnect()
        self.connect()

    def __getstate__(self):
        self.disconnect()
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.connect()
