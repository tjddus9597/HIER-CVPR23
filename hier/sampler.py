import torch
import random
from torch.utils.data.sampler import Sampler
import numpy as np
import collections


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=np.int)
    return labels_to_indices


def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace).tolist()


class UniqueClassSempler(Sampler):
    def __init__(self, labels, m_per_class, rank=0, world_size=1, seed=0):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = labels
        self.label_set = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (len(self.label_set) // self.world_size) * self.m_per_class

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.label_set), generator=g).tolist()
        size = len(self.label_set) // self.world_size
        idx = idx[size * self.rank : size * (self.rank + 1)]
        for i in idx:
            t = self.labels_to_indices[self.label_set[i]]
            idx_list += safe_random_choice(t, self.m_per_class)
        return iter(idx_list)

    def set_epoch(self, epoch):
        self.epoch = epoch


class UniqueClassSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, batch_size, images_per_class=3, rank=0, world_size=1):
        self.targets = targets
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.rank = rank
        self.world_size = world_size
        self.reverse_index, self.ignored = self._build_reverse_index()
        self.epoch = 0

    def __iter__(self):
        num_batches = len(self.targets) // (self.world_size * self.batch_size)
        ret = []
        i = 0
        while num_batches > 0:
            ret.extend(self.sample_batch(i))
            num_batches -= 1
            i += 1
        return iter(ret) 

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self, batch_idx):
        np.random.seed(batch_idx * 10000 + self.epoch)
        num_classes = self.batch_size * self.world_size // self.images_per_class
        replace = num_classes > len(list(set(self.targets)))
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=replace)
        size = num_classes // self.world_size
        sampled_classes = sampled_classes[size * self.rank : size * (self.rank + 1)]

        np.random.seed(random.randint(0, 1e6))
        sampled_indices = []
        for cls_idx in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls_idx], self.images_per_class, replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
class BalancedSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, batch_size, images_per_class=3, world_size=1):
        self.targets = targets
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.world_size = world_size
        self.reverse_index, self.ignored = self._build_reverse_index()

    def __iter__(self):
        num_batches = len(self.targets) // (self.world_size * self.batch_size)
        ret = []
        while num_batches > 0:
            ret.extend(np.random.permutation(self.sample_batch()))
            num_batches -= 1
        return iter(ret) 

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self):
        # Real batch size is self.images_per_class * (self.batch_size // self.images_per_class)
        num_classes = self.batch_size // self.images_per_class
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=False)

        sampled_indices = []
        for cls_idx in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls_idx], self.images_per_class, replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch