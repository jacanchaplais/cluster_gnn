import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from cluster_gnn.models import gnn
from cluster_gnn.data import loader as loadset


ROOT_DIR = '/home/jlc1n20/projects/cluster_gnn/'
MODEL_DIR = ROOT_DIR + 'models/'
LOG_DIR = ROOT_DIR + 'log/'

graph_data = GraphDataModule('/home/jlc1n20/projects/cluster_gnn/data/')
model = gnn.Net(num_hidden=7, dim_embed_edge=128, dim_embed_node=128)
logger = pl.loggers.TensorBoardLogger(MODEL_DIR)
trainer = pl.Trainer(gpus=4, num_nodes=1, max_epochs=30,
                     progress_bar_refresh_rate=100,
                     logger=logger, default_root_dir=MODEL_DIR,
                     accelerator='ddp')
trainer.fit(model, graph_data)
trainer.test(datamodule=graph_data)
