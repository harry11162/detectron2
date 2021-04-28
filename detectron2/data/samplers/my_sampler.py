import torch
import itertools
from torch.utils.data.sampler import Sampler

class VFSPairSampler(Sampler):
    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None, sample_range: int = 3):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self.sample_range = sample_range

    def __iter__(self):
        start = self._rank
        for i in itertools.islice(self._infinite_indices(), start, None, self._world_size):
            yield i
            offset = torch.randint(self.sample_range, ()) + 1
            if torch.randint(2, ()) == 0:
                offset = -offset
            yield i + offset

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)