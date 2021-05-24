import os

import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from cluster_gnn.models import gnn
from cluster_gnn.data import loader

# slurm hack
os.environ["SLURM_JOB_NAME"] = "bash"

ROOT_DIR = '/home/jlc1n20/projects/cluster_gnn/'
MODEL_DIR = ROOT_DIR + '/models/tune/'

def train_gnn(config, data_module, num_epochs=10, num_gpus=4, callbacks=None,
              checkpoint_dir=None):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(), name="", version=".")
    if checkpoint_dir:
        ckpt = pl.utilities.cloud_io.pl_load(
            os.path.join(checkpoint_dir, 'checkpoint'),
            map_location=lambda storage, loc: storage)
        model = gnn.Net._load_model_state(
            checkpoint=ckpt,
            num_hidden=6, dim_embed_edge=64, dim_embed_node=32,
            learn_rate=config['learn_rate'],
            pos_weight=config['pos_weight'])
    else:
        model = gnn.Net(num_hidden=6, dim_embed_edge=64, dim_embed_node=32,
                        learn_rate=config['learn_rate'],
                        pos_weight=config['pos_weight'])
    trainer = pl.Trainer(gpus=num_gpus, num_nodes=1, max_epochs=num_epochs,
                         progress_bar_refresh_rate=0,
                         limit_train_batches=0.1,
                         logger=logger,
                         callbacks=callbacks)
    trainer.fit(model, data_module)
    print('callback metrics are:\n {}'.format(trainer.callback_metrics))

def tune_gnn(data_module, num_samples=10, num_epochs=10, gpus_per_trial=2,
             init_params=None, checkpoint_dir=None):
    config = {
        'learn_rate': tune.loguniform(1e-6, 1e-1),
        'pos_weight': tune.uniform(1.0, 100.0),
        }
    metrics = ['ptl/val_loss', 'ptl/val_accuracy', 'ptl/val_f']
    callbacks = [
        TuneReportCheckpointCallback(
            metrics,
            filename='checkpoint',
            on='validation_end')
        ]
    scheduler = ASHAScheduler(
        time_attr='epoch',
        max_t=num_epochs,
        )
    search_alg = HyperOptSearch(points_to_evaluate=init_params)
    reporter = CLIReporter(
        parameter_columns=[
            'learn_rate',
            'pos_weight',
            ],
        metric_columns=metrics+['epoch'],
        )
    trainable = tune.with_parameters(
        train_gnn,
        data_module=data_module,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        callbacks=callbacks,
        checkpoint_dir=checkpoint_dir,
        )
    analysis = tune.run(
        trainable,
        resources_per_trial={
            'cpu': 1,
            'gpu': gpus_per_trial,
            },
        metric='ptl/val_f',
        mode='max',
        config=config,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=MODEL_DIR,
        verbose=3,
        name='tune_gnn')
    print('Best hp found: ', analysis.best_config)


if __name__ == '__main__':
    # currently can't parallelise while tuning, see:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/7671
    num_gpus = 1 
    cur_best_params = [{
        'learn_rate': 3.75e-5,
        'pos_weight': 21.5,
        }]
    graph_data = loader.GraphDataModule(
        '/home/jlc1n20/projects/cluster_gnn/data/', num_workers=num_gpus)

    tune_gnn(data_module=graph_data, num_samples=18, num_epochs=10,
             gpus_per_trial=num_gpus, init_params=cur_best_params)

