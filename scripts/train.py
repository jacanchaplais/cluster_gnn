import csv

import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from cluster_gnn.models import gnn
from cluster_gnn.data import loader as loadset

NUM_EPOCHS = 30
DIM_EMBED_EDGE = 64
DIM_EMBED_NODE = 32
FINAL_BIAS = False # bias in the classification layer
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 5e-4 # L2 regularisation
MODEL_DIR = '/home/jlc1n20/projects/cluster_gnn/models/'
HANDLE = 'lr14'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gnn.Net(dim_embed_edge=DIM_EMBED_EDGE,
                dim_embed_node=DIM_EMBED_NODE,
                final_bias=FINAL_BIAS).to(device)
optimiser = torch.optim.Adam(model.parameters(),
                             lr=LR, weight_decay=WEIGHT_DECAY)
# use a loss to penalise false negatives
criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(100.0), reduction='mean')
dataset = loadset.EventDataset()
dataset = dataset[0:int(0.9 * dataset.len())] # 90% for training
loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

def train(data):
    optimiser.zero_grad()
    edge_pred = model(data)
    loss = criterion(edge_pred, data.y.view(-1, 1))
    loss.backward()
    optimiser.step()
    return loss.detach().item()

model.train()
log_names = ['Epoch', 'Loss']
for epoch in range(NUM_EPOCHS):
    tot_loss = 0.0
    for num, data in enumerate(tqdm(loader)):
        tot_loss += train(data.to(device))
    av_loss = tot_loss / float(len(loader))
    torch.save(model.state_dict(), MODEL_DIR + HANDLE + '.pt')
    with open(MODEL_DIR + 'loss_' + HANDLE + '.csv', 'a') as lossfile:
        writer = csv.DictWriter(lossfile, fieldnames=log_names)
        writer.writerow({
            'Epoch': f'{epoch}' + '/' + f'{NUM_EPOCHS}',
            'Loss': av_loss
            })
