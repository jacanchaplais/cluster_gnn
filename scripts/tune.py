import os

import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from cluster_gnn.models import gnn
from cluster_gnn.data import loader

ROOT_DIR = '/home/jlc1n20/projects/cluster_gnn/'
MODEL_DIR = ROOT_DIR + '/models/tune/'

def objective(trial):
    prune_metric = 'val_loss'

    learn_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    pos_weight = trial.suggest_float("pos_weight", 1.0, 100.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    graph_data = loader.GraphDataModule(
            '/home/jlc1n20/projects/cluster_gnn/data/',
            num_workers=4)
    model = gnn.Net(num_hidden=7, dim_embed_edge=128, dim_embed_node=128,
                    learn_rate=learn_rate,
                    pos_weight=pos_weight,
                    weight_decay=weight_decay)
    trainer = pl.Trainer(gpus=-1, num_nodes=1, max_epochs=8,
                         logger=True, default_root_dir=MODEL_DIR,
                         checkpoint_callback=False,
                         callbacks=[PyTorchLightningPruningCallback(
                             trial, monitor=prune_metric)],
                         accelerator='ddp')

    hyps = {'lr': learn_rate,
            'pos_weight': pos_weight,
            'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyps)
    trainer.fit(model, graph_data)

    precision = trainer.callback_metrics['val_prec/thresh_0.500'].item()
    recall = trainer.callback_metrics['val_recall/thresh_0.500'].item()

    f1 = 2.0 * precision * recall / (precision + recall)
    return f1

pruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=10)

print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print(" Value: {}".format(trial.value))

print(" Params: ")
for key, val in trial.params.items():
    print("    {}: {}".format(key, value))
