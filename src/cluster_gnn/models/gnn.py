from itertools import chain

import torch
import torchmetrics
import torch_geometric as pyg
import pytorch_lightning as pl

class Interaction(pyg.nn.MessagePassing):
    def __init__(self, in_edge, in_node, out_edge, out_node):
        super(Interaction, self).__init__(
            aggr='add',
            flow="source_to_target")
        self.in_edge = 2 * in_node + in_edge
        self.in_node = in_node + out_edge
        self.mlp_edge = torch.nn.Sequential(
            torch.nn.Linear(self.in_edge, out_edge, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(out_edge, out_edge, bias=True)
        )
        self.mlp_node = torch.nn.Sequential(
            torch.nn.Linear(self.in_node, out_node, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(out_node, out_node, bias=True)
        )

    def forward(self, x, edge_index, edge_attrs):
        return self.propagate(
            x=x,
            edge_index=edge_index,
            edge_attrs=edge_attrs
        )

    def message(self, x_i, x_j, edge_index, edge_attrs):
        recv_send = [x_i, x_j]
        if edge_attrs is not None:
            recv_send.append(edge_attrs)
        recv_send = torch.cat(recv_send, dim=1)
        self.edge_embed = self.mlp_edge(recv_send)
        return self.edge_embed

    def update(self, aggr_out, x):
        node_embed = self.mlp_node(torch.cat([x, aggr_out], dim=1))
        return (self.edge_embed, node_embed)


class Net(pl.LightningModule):
    def __init__(self, dim_node: int = 4, dim_edge: int = 0,
                 dim_embed_edge: int = 64, dim_embed_node: int = 64,
                 dim1_node=None, dim1_edge=None,
                 num_hidden: int = 7, final_bias: bool = True,
                 pos_weight: float = 4.5,
                 learn_rate: float = 1.0e-4,
                 weight_decay: float = 3.0e-5,
                 infer_thresh: float = 0.5):
        super(Net, self).__init__()
        # define the architecture
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_embed_node = dim_embed_node
        self.dim_embed_edge = dim_embed_edge
        self.__expand = lambda inp: list(chain.from_iterable(inp))

        self.dims = { # True if in first layer
            'edge': {
                True: dim1_edge,
                False: self.dim_embed_edge,
                },
            'node': {
                True: dim1_node,
                False: self.dim_embed_node + self.dim_node,
                }
            }

        self.__cat = lambda x1, x2: torch.cat([x1, x2], dim=1)

        self.model = pyg.nn.Sequential('x_init, edge_index, edge_attrs',
            [( # encoder
                Interaction(
                    self.dim_edge, self.dim_node,
                    self.dim_embed_edge, self.dim_embed_node
                    ),
                'x_init, edge_index, edge_attrs -> edge_attrs, x'
                ),
            ]
            + self.__expand([(
                (self.__cat, 'x_init, x -> x_in'),
                (Interaction(
                    self.dims['edge'][i == 0 and dim1_edge != None],
                    self.dims['node'][i == 0 and dim1_node != None],
                    self.dim_embed_edge, self.dim_embed_node),
                'x_in, edge_index, edge_attrs -> edge_attrs, x')
                ) for i in range(num_hidden)])
            )

        self.classify = torch.nn.Linear(dim_embed_edge, 1, bias=final_bias)
        # optimiser args
        self.lr = learn_rate
        self.decay = weight_decay
        # configure the loss
        self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, device=self.device),
                reduction='mean')
        # METRICS:
        # train
        self.train_ACC = torchmetrics.Accuracy(threshold=infer_thresh)
        self.train_F1 = torchmetrics.F1(
                num_classes=1, threshold=infer_thresh)
        # validate
        self.val_ACC = torchmetrics.Accuracy(threshold=infer_thresh)
        self.val_F1 = torchmetrics.F1(
                num_classes=1, threshold=infer_thresh)
        self.val_PR = torchmetrics.BinnedPrecisionRecallCurve(
                num_classes=1, num_thresholds=5)
        # test
        self.test_PR = torchmetrics.BinnedPrecisionRecallCurve(
                num_classes=1)
        self.test_ROC = torchmetrics.ROC(compute_on_step=False)
        self.test_AUC = torchmetrics.AUROC()

    def forward(self, data, sigmoid=True):
        node_attrs, edge_attrs = data.x, data.edge_attr
        edge_attrs, node_attrs = self.model(node_attrs, data.edge_index,
                                            edge_attrs)
        pred = self.classify(edge_attrs)
        if sigmoid:
            pred = torch.sigmoid(pred)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.decay
            )
        return optimizer

    def _train_av_loss(self, outputs):
        return torch.stack([x['loss'] for x in outputs]).mean()

    def _val_av_loss(self, losses):
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        edge_pred = self(batch, sigmoid=False)
        loss = self.criterion(edge_pred, batch.y.view(-1, 1))
        return {'loss': loss,
                'preds': torch.sigmoid(edge_pred),
                'target': batch.y.view(-1, 1).int()}

    def training_step_end(self, outputs):
        self.train_ACC(outputs['preds'], outputs['target'])
        self.train_F1(outputs['preds'], outputs['target'])
        self.log('ptl/train_loss', outputs['loss'], on_step=True)
        return outputs['loss']

    def training_epoch_end(self, outputs):
        self.log('ptl/train_loss', self._train_av_loss(outputs))
        self.log('ptl/train_accuracy', self.train_ACC.compute())
        self.log('ptl/train_f', self.train_F1.compute())

    def validation_step(self, batch, batch_idx):
        edge_pred = self(batch, sigmoid=False)
        loss = self.criterion(edge_pred, batch.y.view(-1, 1))
        return {'loss': loss,
                'preds': torch.sigmoid(edge_pred),
                'target': batch.y.view(-1, 1).int()}

    def validation_step_end(self, outputs):
        self.val_ACC(outputs['preds'], outputs['target'])
        self.val_F1(outputs['preds'], outputs['target'])
        self.val_PR(outputs['preds'], outputs['target'])
        self.log('ptl/val_loss', outputs['loss'], on_step=True)
        return outputs['loss']

    def validation_epoch_end(self, outputs):
        metrics = {
            'ptl/val_loss': self._val_av_loss(outputs),
            'ptl/val_accuracy': self.val_ACC.compute(),
            'ptl/val_f': self.val_F1.compute(),
            }
        self.log_dict(metrics, sync_dist=True)
        prec, recall, thresh = self.val_PR.compute()
        for i, t in enumerate(thresh):
            self.log(f'ptl/val_prec_thresh_{t:.3f}', prec[i])
            self.log(f'ptl/val_recall_thresh_{t:.3f}', recall[i])

    def test_step(self, batch, batch_idx):
        edge_pred = self(batch, sigmoid=False)
        loss = self.criterion(edge_pred, batch.y.view(-1, 1))
        preds = torch.sigmoid(edge_pred)
        target = batch.y.view(-1, 1).int()
        self.test_ROC(preds, target)
        self.test_AUC(preds, target)
        # self.test_PR(preds, target)
        return {'loss': loss,
                'preds': preds,
                'target': target}

    def test_epoch_end(self, outputs):
        # prec, rec, thresh = self.test_PR.compute()
        roc = self.test_ROC.compute()
        auc = self.test_AUC.compute()
        out_dir = '/home/jlc1n20/projects/cluster_gnn/models/big/'
        # torch.save(prec, out_dir + 'prec.pt')
        # torch.save(rec, out_dir + 'rec.pt')
        # torch.save(thresh, out_dir + 'thresh.pt')
        torch.save(auc, out_dir + 'auc.pt')
        torch.save(roc, out_dir + 'roc.pt')
        # self.log('ptl/test_roc', self.test_ROC.compute())
        # self.log('ptl/test_auc', self.test_AUC.compute())
