import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from cluster_gnn.models import gnn
from cluster_gnn.data import loader

ROOT_DIR = '/home/jlc1n20/projects/cluster_gnn/'
MODEL_DIR = ROOT_DIR + '/models/tune/'


def train_gnn(config, num_epochs=10, num_gpus=4):
    model = gnn.Net(num_hidden=7, dim_embed_edge=64, dim_embed_node=32,
                    learn_rate=config['learn_rate'],
                    pos_weight=config['pos_weight'])

    graph_data = loader.GraphDataModule(
        '/home/jlc1n20/projects/cluster_gnn/data/')

    logger = pl.loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(), name="", version=".")

    report_callback = TuneReportCallback({
            'loss': 'loss/val_epoch',
            'mean_f1': 'f1/val_epoch'
            },
        on='validation_end')

    trainer = pl.Trainer(gpus=num_gpus, num_nodes=1, max_epochs=num_epochs,
                         progress_bar_refresh_rate=0,
                         logger=logger,
                         callbacks=[report_callback])
    trainer.fit(model, graph_data)

def tune_gnn(num_samples=10, num_epochs=10, gpus_per_trial=2):
    config = {
        'learn_rate': tune.loguniform(1e-6, 1e-1),
        'pos_weight': tune.uniform(1.0, 120.0),
        }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
        )

    reporter = CLIReporter(
        parameter_columns=[
            'learn_rate',
            'pos_weight',
            ],
        )

    analysis = tune.run(
        tune.with_parameters(
            train_gnn,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
            ),
        resources_per_trial={
            'cpu': 2,
            'gpu': gpus_per_trial,
            },
        metric='mean_f1',
        mode='max',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=MODEL_DIR,
        name='tune_gnn')
    print('Best hp found: ', analysis.best_config)

tune_gnn(num_samples=10, num_epochs=10, gpus_per_trial=4)
