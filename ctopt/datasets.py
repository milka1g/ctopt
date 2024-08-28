"""
@File    :   con_dataset.py
@Time    :   2024/08/27 14:42:05
@Author  :   Nikola Milicevic
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

from collections import defaultdict
from torch.utils.data import Dataset
import torch
import random
import numpy as np


class ContrastiveDataset(Dataset):
    """Dataset that returns an item and n_views of it based on target."""

    def __init__(self, data, target, n_views=2):
        self.data = np.array(data)
        self.target = np.array(target)
        unique_targets = np.unique(target)
        self.target_to_group = defaultdict(list)
        for ut in unique_targets:
            self.target_to_group[ut] = [
                ind for ind, element in enumerate((target == ut)) if element
            ]
        self.n_views = n_views

    def __getitem__(self, index):
        """Retrieves one training sample for batch formation with n_views

        Args:
            index: index of an element in dataset.

        Returns:
            A tuple of n_views of positives (same cell type)
        """
        anchor_target = self.target[index]
        anchor_sample = torch.tensor(self.data[index], dtype=torch.float)

        # Collect n_views of postiives from group with same label
        same_group = self.target_to_group[anchor_target]

        random_idx = index
        views = [anchor_sample]
        for _ in range(self.n_views - 1):
            while random_idx == index:
                random_idx = random.choice(same_group)
            views.append(torch.tensor(self.data[random_idx], dtype=torch.float))

        return (
            (views, anchor_target)
            if self.n_views > 1
            else (anchor_sample, anchor_target)
        )

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape


class EmbDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
