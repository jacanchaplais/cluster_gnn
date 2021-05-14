import torch
from torch_geometric.data import Data, Dataset
import h5py
import numpy as np
from numba import jit


# TODO: add processing, download, maybe env var for data dir
class EventDataset(Dataset):
    def __init__(self,
                 data_dir: str = './data/',
                 transform=None,
                 pre_transform=None):
        super(EventDataset, self).__init__(None, transform, pre_transform)
        self.root_dir = data_dir
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
    
    @jit(forceobj=True)
    def _get_edges(self, num_nodes):
        """Returns COO formatted graph edges for
        full connected graph of given number of nodes.
        type: (2, num_nodes * (num_nodes - 1)) dim array
        """
        nodes = np.arange(num_nodes, dtype=np.int64)
        edge_idx = np.vstack((
            np.repeat(nodes, num_nodes),
            np.repeat(nodes, num_nodes).reshape(-1, num_nodes).T.flatten()
        ))
        # removing self-loops
        mask = edge_idx[0] != edge_idx[1]
        edge_idx = edge_idx[:, mask]
        return edge_idx
    
    def get(self, idx):
        with h5py.File(self.root_dir + '/processed/events.hdf5', 'r') as f:
            # LOAD DATA:
            evt = f['wboson'][f'event_{idx:06}']
            num_nodes = evt.attrs['num_pcls']
            pmu = torch.from_numpy(evt['pmu'][...]) # 4-momentum for nodes
            if 'edges' in evt:
                edge_idx = evt['edges'][...].astype(np.int64)
            else:
                edge_idx = self._get_edges(num_nodes)
            edge_idx = torch.from_numpy(edge_idx).long()
            pdg = torch.from_numpy(evt['pdg'][...]) # PDG for posterity
            
            # CONSTRUCT EDGE LABELS:
            is_from_W = evt['is_from_W'][...]
            # node pairs bool labelled for all edges
            is_from_W = is_from_W[edge_idx]
            # reduce => True if both nodes True
            edge_labels = np.bitwise_and.reduce(is_from_W, axis=0)
            edge_labels = torch.from_numpy(edge_labels).float()
            
            # RETURN GRAPH
            return Data(x=pmu, edge_index=edge_idx,
                        y=edge_labels, pdg=pdg)

class GraphDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = './data/',
                 splits: list[float] = [0.9, 0.05, 0.05],
                 batch_size: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        if sum(splits) <= 1.0:
            self.splits = splits
        else:
            raise ArithmeticError('Sum of splits must not exceed 1.0')

        def setup(self, stage: Optional[str] = None):
            # stage could be fit, test
            graph_set = EventDataset(self.data_dir)
            num_graphs = graph_set.len()
            splits = [round(split * float(num_graphs)) in self.splits]
            self.train, self.val, self.test = torch.utils.data.random_split(
                    graph_set, splits)

        def train_dataloader(self):
            return DataLoader(self.train, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.test, batch_size=self.batch_size)
