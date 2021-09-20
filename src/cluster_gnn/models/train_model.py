import os

import pytorch_lightning as pl
import yaml
import click

from cluster_gnn import ROOT_DIR
from cluster_gnn.models import gnn
from cluster_gnn.data import loader


# TODO: add a func for finding latest checkpoint
# def latest_ckpt(model_dir):
#     ckpt_dirs = [f.path for f in os.scandir(model_dir) if f.is_dir()]
#     ckpt_dirs.sort(key=lambda e: int(e.replace(model_dir + 'version_', '')))
#     latest_dir = ckpt_dirs[-1]
#     ckpt_path = latest_dir + '/checkpoints/'

@click.command()
@click.option('-c', '--config',
              type=click.File(),
              default=ROOT_DIR+'/configs/default.yml'
              )
def train_model(config):
    SETTINGS = yaml.safe_load(config)
    model_dir = ROOT_DIR + '/models/' + SETTINGS['name'] + '/'
    ckpt_path = None
    if SETTINGS['optim']['ckpt_path'] != '':
        ckpt_path = SETTINGS['optim']['ckpt_path']

    # configure data loaders, GNN, and trainer settings
    graph_data = loader.GraphDataModule(
        str(SETTINGS['data']['dir']).replace('%ROOT%', ROOT_DIR),
        splits=SETTINGS['data']['splits'],
        batch_size=int(SETTINGS['data']['batch_size']),
        edge_weight=bool(SETTINGS['data']['edge_weight']),
        use_charge=bool(SETTINGS['data']['pcl_props']['charge']),
        num_workers=int(SETTINGS['device']['num_workers']),
        knn=int(SETTINGS['data']['knn']),
        )
    model = gnn.Net(
        num_hidden=int(SETTINGS['arch']['num_hidden']),
        dim1_edge=int(SETTINGS['arch']['dim_embed_edge_l1']),
        dim1_node=int(SETTINGS['arch']['dim_embed_node_l1']),
        dim_embed_edge=int(SETTINGS['arch']['dim_embed_edge']),
        dim_embed_node=int(SETTINGS['arch']['dim_embed_node']),
        final_bias=bool(SETTINGS['arch']['final_bias']),
        learn_rate=float(SETTINGS['optim']['learn_rate']),
        weight_decay=float(SETTINGS['optim']['weight_decay']),
        pos_weight=float(SETTINGS['loss']['pos_weight']),
        use_charge=bool(SETTINGS['data']['pcl_props']['charge']),
        )
    trainer_kwargs = dict(
        gpus=int(SETTINGS['device']['num_gpus']),
        num_nodes=int(SETTINGS['device']['num_nodes']),
        max_epochs=int(SETTINGS['optim']['num_epochs']),
        progress_bar_refresh_rate=100,
        logger=pl.loggers.TensorBoardLogger(model_dir),
        default_root_dir=model_dir,
        accelerator=str(SETTINGS['device']['accelerator']),
        )

    # initiate training, resuming from checkpoint if specified
    if os.path.exists(model_dir) and ckpt_path != None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('Checkpoint path does not exist.')
        trainer = pl.Trainer(
            resume_from_checkpoint=ckpt_path, **trainer_kwargs)
    else:
        trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, graph_data)

if __name__ == '__main__':
    train_model()
