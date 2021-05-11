import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from cluster_gnn.models import gnn
from cluster_gnn.data import loader as loadset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gnn.Net().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
# use a loss to penalise false negatives
criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(100.0), reduction='sum')
dataset = loadset.EventDataset()
dataset = dataset[0:int(0.9 * dataset.len())] # 90% for training
loader = DataLoader(dataset, shuffle=True, batch_size=1)

for epoch in range(15):
    print(f'Epoch {epoch}:')
    for num, data in enumerate(tqdm(loader)):
        data = data.to(device)
        optimiser.zero_grad()
        edge_pred = torch.nn.Sigmoid(model(data))
        loss = criterion(edge_pred, data.y.view(-1, 1))
        loss.backward()
        optimiser.step()

    torch.save(model, '/home/jlc1n20/projects/cluster_gnn/models/edge.pt')

