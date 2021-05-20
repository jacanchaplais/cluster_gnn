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
                 dim_embed_edge: int = 64, dim_embed_node: int = 32,
                 num_hidden: int = 3, final_bias: bool = False,
                 pos_weight: float = 80.0,
                 learn_rate: float = 1e-4, weight_decay: float = 5e-4,
                 infer_thresh: float = 0.5):
        super(Net, self).__init__()
        # define the architecture
        self.encode = Interaction(dim_edge, dim_node,
                                  dim_embed_edge, dim_embed_node)
        self.message = pyg.nn.Sequential('x, edge_index, edge_attrs', [
            (Interaction(dim_embed_edge, dim_embed_node,
                         dim_embed_edge, dim_embed_node),
             'x, edge_index, edge_attrs -> edge_attrs, x')
             for i in range(num_hidden)
             ])
        self.classify = torch.nn.Linear(dim_embed_edge, 1, bias=final_bias)
        # optimiser args
        self.lr = learn_rate
        self.decay = weight_decay
        # configure the loss
        self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, device=self.device),
                reduction='mean')
        # add metrics
        self.train_ACC = torchmetrics.Accuracy(threshold=infer_thresh)
        self.train_F1 = torchmetrics.F1(threshold=infer_thresh)
        self.val_ACC = torchmetrics.Accuracy(threshold=infer_thresh)
        self.val_F1 = torchmetrics.F1(
                num_classes=1, threshold=infer_thresh)
        self.val_PR = torchmetrics.BinnedPrecisionRecallCurve(
                num_classes=1, num_thresholds=5)

    def forward(self, data, sigmoid=True):
        node_attrs, edge_attrs = data.x, data.edge_attr
        edge_attrs, node_attrs = self.encode(node_attrs, data.edge_index,
                                             edge_attrs)
        edge_attrs, node_attrs = self.message(node_attrs, data.edge_index,
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
        self.log('loss/train_step', outputs['loss'], on_step=True)
        self.log('acc/train_step', self.train_ACC, on_step=True)
        self.log('f1/train_step', self.train_F1, on_step=True)
        return outputs['loss']

    def training_epoch_end(self, outputs):
        self.log('loss/train_epoch', self._train_av_loss(outputs))
        self.log('acc/train_epoch', self.train_ACC.compute())
        self.log('f1/train_epoch', self.train_F1.compute())

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
        self.log('loss/val_step', outputs['loss'], on_step=True)
        self.log('acc/val_step', self.val_ACC, on_step=True)
        return outputs['loss']

    def validation_epoch_end(self, outputs):
        self.log('loss/val_epoch', self._val_av_loss(outputs))
        self.log('acc/val_epoch', self.val_ACC.compute())
        self.log('f1/val_epoch', self.val_F1.compute())
        prec, recall, thresh = self.val_PR.compute()
        for i, t in enumerate(thresh):
            self.log(f'prec_val_epoch/thresh_{t:.3f}', prec[i])
            self.log(f'recall_val_epoch/thresh_{t:.3f}', recall[i])

