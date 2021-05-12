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
            num_workers=1)
    start = end

model = gnn.Net()
logger = pl.loggers.TensorBoardLogger(LOG_DIR)
profiler = pl.profiler.AdvancedProfiler(dirpath=LOG_DIR, filename='prof.txt')
trainer = pl.Trainer(gpus=2, max_epochs=1, progress_bar_refresh_rate=100,
                     logger=logger, default_root_dir=MODEL_DIR,
                     accelerator='ddp', profiler=profiler)
trainer.fit(model, loaders['train'], loaders['val'])
# trainer.test(test_dataloaders=loaders['test'])
