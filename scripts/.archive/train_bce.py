import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from cluster_gnn.models import gnn
from cluster_gnn.data import loader as loadset
from cluster_gnn.loss.classify import bce_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gnn.Net().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
# use a loss to penalise false negatives
dataset = loadset.EventDataset()
dataset = dataset[0:int(0.9 * dataset.len())] # 90% for training
loader = DataLoader(dataset, shuffle=True, batch_size=1)

for epoch in range(15):
    print(f'Epoch {epoch}:')
    for num, data in enumerate(tqdm(loader)):
        data = data.to(device)
        optimiser.zero_grad()
        edge_pred = torch.sigmoid(model(data))
        label = data.y.to(device)
        loss = bce_loss(edge_pred.squeeze(1), label,
                        false_neg_weight=1000.0)
        loss.backward()
        optimiser.step()

    torch.save(model, '/home/jlc1n20/projects/cluster_gnn/models/edge_bce.pt')

