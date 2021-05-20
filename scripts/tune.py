import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from cluster_gnn.models import gnn
from cluster_gnn.data import loader

ROOT_DIR = '/home/jlc1n20/projects/cluster_gnn/'
MODEL_DIR = ROOT_DIR + '/models/tune/'

graph_data = loader.GraphDataModule(
    '/home/jlc1n20/projects/cluster_gnn/data/', num_workers=4)

def train_gnn(config, num_epochs=10, num_gpus=4, checkpoint_dir=None):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(), name="", version=".")
    report_callback = TuneReportCheckpointCallback(
        metrics={
            'loss': 'loss/val_epoch',
            'mean_f1': 'f1/val_epoch'
        },
        filename='checkpoint',
        on='validation_end')
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
                         logger=logger,
                         callbacks=[report_callback])
    trainer.fit(model, graph_data)

def tune_gnn(num_samples=10, num_epochs=10, gpus_per_trial=2,
             init_params=None):
    config = {
        'learn_rate': tune.loguniform(1e-6, 1e-1),
        'pos_weight': tune.uniform(1.0, 100.0),
        }
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
        )
    analysis = tune.run(
        tune.with_parameters(
            train_gnn,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
            ),
        resources_per_trial={
            'cpu': 1,
            'gpu': gpus_per_trial,
            },
        metric='mean_f1',
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

cur_best_params = [{
    'learn_rate': 3.75e-5,
    'pos_weight': 21.5,
    }]

tune_gnn(num_samples=18, num_epochs=10, gpus_per_trial=4,
         init_params=cur_best_params)
