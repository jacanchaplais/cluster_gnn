from itertools import accumulate

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix as sps_to_edge
import pytorch_lightning as pl
import h5py
import numpy as np
from numba import jit
import scipy.sparse as sps

from cluster_gnn.features import build_features as bf

# TODO: add processing, download, maybe env var for data dir
class EventDataset(Dataset):
    def __init__(self,
                 data_dir: str = './data/',
                 knn: int = 0,
                 edge_weight: bool = False,
                 transform=None,
                 pre_transform=None):
        super(EventDataset, self).__init__(None, transform, pre_transform)
        self.root_dir = data_dir
        self.knn = knn
        self.edge_weight = edge_weight
        with h5py.File(self.root_dir + '/processed/events.hdf5', 'r') as f:
            self.length = f['wboson'].attrs['num_evts']
        
    @property
    def raw_file_names(self):
        return [self.root_dir + '/external/wboson.txt',
                self.root_dir + '/external/qstar.txt']
    
    @property
    def processed_file_names(self):
        return [self.root_dir + '/processed/events.hdf5']
    
    def len(self):
        return self.length
    
    # @jit(forceobj=True)
    def _get_edges(self, pmu):
        """Returns COO formatted graph edges for
        fully connected graph of given number of nodes.
        type: (2, num_nodes * (num_nodes - 1)) dim array
        """
        dtype = np.float64
        if self.knn == 0:
            adj = bf.fc_adj(num_nodes=len(pmu), dtype=dtype)
        else:
            adj = bf.knn_adj(bf.deltaR_aff(pmu), k=self.knn, dtype=dtype,
                             weighted=self.edge_weight)
        edge_idx = sps_to_edge(sps.coo_matrix(adj)) # to coo formatted edges
        return edge_idx
    
    def get(self, idx):
        with h5py.File(self.root_dir + '/processed/events.hdf5', 'r') as f:
            # LOAD DATA:
            evt = f['wboson'][f'event_{idx:06}']
            num_nodes = evt.attrs['num_pcls']
            pmu = evt['pmu'][...]
            edge_idx, edge_weight = self._get_edges(pmu)
            if not self.edge_weight:
                edge_weight = None
            pmu = torch.from_numpy(pmu)
            pdg = torch.from_numpy(evt['pdg'][...]) # PDG for posterity
            
            # CONSTRUCT EDGE LABELS:
            is_from_W = evt['is_from_W'][...]
            # node pairs bool labelled for all edges
            is_from_W = is_from_W[edge_idx]
            # reduce => True if both nodes True
            edge_labels = np.bitwise_and.reduce(is_from_W, axis=0)
            edge_labels = torch.from_numpy(edge_labels).float()

            # RETURN GRAPH
            return Data(x=pmu, edge_index=edge_idx, edge_attr=edge_weight,
                        y=edge_labels, pdg=pdg)

class GraphDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = './data/',
                 splits = {'train': 0.9, 'val': 0.05, 'test': 0.05},
                 batch_size: int = 1,
                 num_workers: int = 1,
                 edge_weight: bool = False,
                 knn: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.knn = knn
        self.edge_weight = edge_weight
        if sum(splits.values()) <= 1.0:
            self.splits = splits
        elif 'train' not in splits or 'val' not in splits:
            raise ValueError('Splits must contain both train and val')
        else:
            raise ArithmeticError('Sum of splits must not exceed 1.0')

    def setup(self, stage=None):
        # stage could be fit, test
        graph_set = EventDataset(self.data_dir, self.knn, self.edge_weight)
        num_graphs = graph_set.len()
        start_idx = 0
        for key, split in self.splits.items():
            stop_idx = start_idx + round(split * float(num_graphs))
            setattr(self, 'graphs_' + key, graph_set[start_idx:stop_idx])
            start_idx = stop_idx

    def train_dataloader(self):
        return DataLoader(self.graphs_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.graphs_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.graphs_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)
