import os

import pytorch_lightning as pl
import pyyaml

from cluster_gnn import ROOT_DIR
from cluster_gnn.models import gnn
from cluster_gnn.data import loader


# TODO: add a func for finding latest checkpoint
# def latest_ckpt(model_dir):
#     ckpt_dirs = [f.path for f in os.scandir(model_dir) if f.is_dir()]
#     ckpt_dirs.sort(key=lambda e: int(e.replace(model_dir + 'version_', '')))
#     latest_dir = ckpt_dirs[-1]
#     ckpt_path = latest_dir + '/checkpoints/'


def train_model(hparams, data_module, model_dir, num_epochs=30,
                ckpt_path=None):
    model = gnn.Net(**hparams)
    logger = pl.loggers.TensorBoardLogger(model_dir)
    trainer_kwargs = dict(
        gpus=-1,
        num_nodes=1,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=100,
        logger=logger,
        default_root_dir=model_dir,
        accelerator='ddp',
        )
    if os.path.exists(model_dir) and ckpt_path != None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('Checkpoint path does not exist.')
        trainer = pl.Trainer(
            resume_from_checkpoint=ckpt_path, **trainer_kwargs)
    else:
        trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    MODEL_DIR = ROOT_DIR + 'models/concat/'

    graph_data = loader.GraphDataModule(
        ROOT_DIR + '/data/',
        num_workers=16,
        knn=0,
        )
    config = {
        'num_hidden': 7,
        'dim_embed_edge': 128,
        'dim_embed_node': 128,
        'learn_rate': 1e-4,
        'weight_decay': 5e-5,
        'pos_weight': 1.0,
        'final_bias': True,
        }
    train_model(
        hparams=config,
        data_module=graph_data,
        model_dir=MODEL_DIR,
        num_epochs=40,
        )
