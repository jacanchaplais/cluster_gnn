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
        # self.train_AUC = torchmetrics.AUROC()
        self.val_ACC = torchmetrics.Accuracy(threshold=infer_thresh)
        self.val_PR = torchmetrics.BinnedPrecisionRecallCurve(
                num_classes=1, num_thresholds=5)
        self.test_PR = torchmetrics.BinnedPrecisionRecallCurve(
                num_classes=1, num_thresholds=5)
        # self.test_ACC = torchmetrics.Accuracy(threshold=infer_thresh)
        # self.test_ROC = torchmetrics.ROC()
        # self.test_AUC = torchmetrics.AUROC()

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

    def training_step(self, batch, batch_idx):
        edge_pred = self(batch, sigmoid=False)
        loss = self.criterion(edge_pred, batch.y.view(-1, 1))
        self.log('train_loss', loss, logger=True)
        return {'loss': loss,
                'preds': torch.sigmoid(edge_pred),
                'target': batch.y.view(-1, 1).int()}

    def training_step_end(self, outputs):
        self.train_ACC(outputs['preds'], outputs['target'])
        self.log('train_acc', self.train_ACC, prog_bar=True,
                 logger=True)
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        edge_pred = self(batch, sigmoid=False)
        loss = self.criterion(edge_pred, batch.y.view(-1, 1))
        return {'loss': loss,
                'preds': torch.sigmoid(edge_pred),
                'target': batch.y.view(-1, 1).int()}

    def validation_step_end(self, outputs):
        self.val_ACC(outputs['preds'], outputs['target'])
        self.log('val_acc', self.val_ACC, logger=True)
        # self.val_ROC(outputs['preds'], outputs['target'])
        # self.log('val_roc', self.val_ROC, logger=True)
        # self.val_AUC(outputs['preds'], outputs['target'])
        # self.log('val_auc', self.val_AUC, logger=True)
        self.log('val_loss', outputs['loss'], logger=True)
        prec, recall, thresh = self.val_PR(outputs['preds'], outputs['target'])
        for i, t in enumerate(thresh):
            self.log(f'val_prec/thresh_{t:.3f}', prec[i], logger=True)
            self.log(f'val_recall/thresh_{t:.3f}', recall[i], logger=True)
        mid_idx = (len(prec) // 2) + 1
        precision = prec[mid_idx]
        recall = rec[mid_idx]
        f1 = 2.0 * precision * recall / (precision + recall)
        self.log('val_f1', f1, logger=True)
        return outputs['loss']

    def test_step(self, batch, batch_idx):
        edge_pred = self(batch, sigmoid=False)
        loss = self.criterion(edge_pred, batch.y.view(-1, 1))
        return {'loss': loss,
                'preds': torch.sigmoid(edge_pred),
                'target': batch.y.view(-1, 1).int()}

    def test_epoch_end(self, outputs):
        self.test_ACC(outputs['preds'], outputs['target'])
        self.log('test_acc', self.test_ACC, logger=True)
        prec, recall, thresh = self.test_PR(outputs['preds'], outputs['target'])
        for i, t in enumerate(thresh):
            self.log(f'test_prec/thresh_{t:.3f}', prec[i], logger=True)
            self.log(f'test_recall/thresh_{t:.3f}', recall[i], logger=True)
        return outputs['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.decay
            )
        return optimizer
