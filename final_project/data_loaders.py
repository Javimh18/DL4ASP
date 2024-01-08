import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import RandomSampler

class FSSDDataLoader(DataLoader):
    """
    FSSD Dataloader for batching the Dataset
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.subset = dataset.subset
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        idx_full = np.arange(self.n_samples)
        if shuffle != True:
            self.sampler = RandomSampler(idx_full)
        else:
            self.sampler = None
        
        super(FSSDDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)
