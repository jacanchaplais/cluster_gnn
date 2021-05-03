import os
import logging
from itertools import islice
from math import inf

import torch
from torch.utils.data import Dataset

class EventDataset(Dataset):
    """PyTorch dataset for a full collision event.
    """
    def __init__(self, data, num_pts=-1):
        self.data = data
        if num_pts < 0:
            self.num_pts = len(data['num_pcls'])
        elif num_pts > len(data['num_pcls']):
            logging.warn(
                'num_pts larger than total available - using all data.')
            self.num_pts = len(data['num_pcls'])
        else:
            self.num_pts = num_pts

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}
