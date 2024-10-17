import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np

class FcgSampler(Sampler):
    def __init__(self, labels, n_shots, class_per_iter, iterations):
        self.n_shots = n_shots
        self.class_per_iter = class_per_iter
        self.iterations = iterations
        self.labels = labels

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        # 15, 20
        self.indexes = torch.tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)

        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1
            
    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.n_shots
        cpi = self.class_per_iter

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    
    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations