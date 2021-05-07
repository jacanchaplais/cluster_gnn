import torch
from tqdm import tqdm

from cluster_gnn.models import gnn
from cluster_gnn.data import loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gnn.Net().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()
dataset = loader.EventDataset()

for epoch in range(15):
    print(f'Epoch {epoch}')
    for evt_num in tqdm(range(int(0.9 * dataset.len))):
        data = dataset.get(evt_num).to(device)
        optimiser.zero_grad()
        edge_pred = model(data)
        loss = loss_fn(edge_pred.squeeze(1), data.y)
        loss.backward()
        optimiser.step()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss,
        },
        '/home/jlc1n20/projects/cluster_gnn/models/edge.pt')

