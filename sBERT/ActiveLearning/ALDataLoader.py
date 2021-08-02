import numpy as np
import torch

class ALDataLoaderIterator:
    def __init__(self, dataset, selected, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.nonzero(selected)[0]
        if shuffle:
            self.indices = np.random.permutation(self.indices)
        self.i = 0

    def __next__(self):
        if self.i >= len(self.indices):
            raise StopIteration

        first_idx = self.i
        next_idx = min(self.i + self.batch_size, len(self.indices))
        assert next_idx > first_idx
        batch = {
            key: [self.dataset[self.indices[j].item()][key] for j in range(first_idx, next_idx)] for
            key in self.dataset[0].keys()}
        for key in self.dataset[0].keys():
            if type(batch[key][0]) == torch.Tensor:
                batch[key] = torch.cat(batch[key])
        self.i = next_idx
        return batch


class ALDataLoader:
    def __init__(self, dataset, batch_size, shuffle_train, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.n = len(dataset)
        self.selected = np.full((self.n,), False)
        self.rng = np.random.default_rng(seed=seed)
        self.training = True
        self.selection_indices = None
        self.labels = np.array([item['label'] for item in self.dataset])

    def select_indices(self, indices):
        assert not np.any(self.selected[indices])
        self.selected[indices] = True

    def select_k_at_random(self, k, ensure=None):
        mask = ~self.selected
        if ensure is not None:
            mask = mask & self.labels == ensure
        not_selected = np.nonzero(mask)[0]
        assert k <= len(not_selected)
        indices = self.rng.choice(not_selected, k, replace=False)
        self.select_indices(indices)

    def train(self):
        self.training = True
        self.selection_indices = None

    def selection(self):
        self.training = False

    def __iter__(self):
        if self.training:
            return ALDataLoaderIterator(self.dataset, self.selected, self.batch_size,
                                        self.shuffle_train)

        iterator = ALDataLoaderIterator(self.dataset, ~self.selected, self.batch_size,
                                        shuffle=False)
        self.selection_indices = iterator.indices
        return iterator

    def __len__(self):
        if 'warning_issued' not in dir(self):
            print('Warning: called ALDataLoader::__len__ but it has variable length. Returning '
                  'total possible length.')
            self.warning_issued = True
        return (len(self.dataset) - 1) // self.batch_size + 1
