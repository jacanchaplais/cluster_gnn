import os
import math

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

TUNE_NAME = 'pop'
ROOT_DIR = os.path.expanduser('~/projects/cluster_gnn/')
MODEL_DIR = ROOT_DIR + '/models/tune/'

def train_gnn(config, data_module, num_epochs=10, num_gpus=4, callbacks=None,
              checkpoint_dir=None):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(), name="", version=".")
    trainer = pl.Trainer(
        gpus=math.ceil(num_gpus),
        num_nodes=1,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=0,
        logger=logger,
        callbacks=callbacks,
        accelerator='ddp'
        )
    kwargs = {
        'num_hidden': 7,
        'dim_embed_edge': 64,
        'dim_embed_node': 64,
        'dim1_edge': 128, # dim of first embedded edge layer
        'dim1_node': 128, # dim of first embedded node layer
        'learn_rate': config['learn_rate'],
        'weight_decay': config['weight_decay'],
        'pos_weight': config['pos_weight'],
        'final_bias': True,
        }
    if checkpoint_dir:
        ckpt = pl.utilities.cloud_io.load(
            os.path.join(checkpoint_dir, 'checkpoint'),
            map_location=lambda storage, loc: storage
            )
        model = gnn.Net._load_model_state(checkpoint=ckpt, **kwargs)
        trainer.current_epoch = ckpt['epoch']
    else:
        model = gnn.Net(**kwargs)
    trainer.fit(model, data_module)

def tune_gnn(data_module, num_samples=10, num_epochs=10, gpus_per_trial=1,
             init_params=None, checkpoint_dir=None):
    config = {
        'learn_rate': 1.8e-4, # tune.loguniform(1e-5, 1e-3),
        'pos_weight': 5.0, # tune.randn(5.0, 1.5),
        'weight_decay': 1.9e-4, # tune.loguniform(1e-5, 1e-3),
        }
    metrics = {
        'val_loss': 'ptl/val_loss',
        'val_acc': 'ptl/val_accuracy',
        'val_f': 'ptl/val_f',
        }
    opt_metric = 'val_f'
    opt_mode = 'max'
    opt_tstep = 'training_iteration'
    callbacks = [
        TuneReportCheckpointCallback(
            metrics,
            filename='checkpoint',
            on='validation_end')
        ]
    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            'learn_rate': tune.loguniform(1e-5, 1e-3),
            'pos_weight': tune.randn(5.0, 1.5),
            'weight_decay': tune.loguniform(1e-5, 1e-3),
            },
        )
    search_alg = HyperOptSearch(points_to_evaluate=init_params)
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=list(metrics.values())+[opt_tstep],
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
        metric=opt_metric,
        mode=opt_mode,
        config=config,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=MODEL_DIR,
        verbose=3,
        # resume=True,
        name=TUNE_NAME,
        )
    print('Best hp found: ', analysis.best_config)


if __name__ == '__main__':
    # currently can't parallelise while tuning, see:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/7671
    graph_data = loader.GraphDataModule(ROOT_DIR + '/data/')

    tune_gnn(
        data_module=graph_data,
        num_samples=3,
        num_epochs=26,
        gpus_per_trial=2,
        )
