import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from cluster_gnn.models import gnn
from cluster_gnn.data import loader as loadset


ROOT_DIR = '/home/jlc1n20/projects/cluster_gnn/'
MODEL_DIR = ROOT_DIR + 'models/'
LOG_DIR = ROOT_DIR + 'log/'

# SETTING UP THE LOADERS
SPLITS = {'train': 0.9, 'test': 0.05, 'val': 0.05}
loaders = {}
dataset = loadset.EventDataset()
num_graphs = dataset.len()
start = 0.0 # fractional location at start of each split
for key, frac in SPLITS.items():
    end = start + frac
    slc = slice(int(start * num_graphs), int(end * dataset.len()))
    loaders[key] = DataLoader(
            dataset[slc],
            shuffle=(True if key == 'train' else False),
            batch_size=2,
            num_workers=4)
    start = end

model = gnn.Net(num_hidden=7, dim_embed_edge=128, dim_embed_node=128)
logger = pl.loggers.TensorBoardLogger(LOG_DIR)
trainer = pl.Trainer(gpus=4, num_nodes=1, max_epochs=30,
                     progress_bar_refresh_rate=100,
                     logger=logger, default_root_dir=MODEL_DIR,
                     accelerator='ddp')
trainer.fit(model, loaders['train'], loaders['val'])
# trainer.test(test_dataloaders=loaders['test'])
