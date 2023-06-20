import torch 
import numpy as np
import pandas as pd
import random
import json
from matplotlib import pyplot as plt
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,Sequential,GATConv,GATv2Conv,Linear,Sequential

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"



class GAT(torch.nn.Module):
    def __init__(self, seed,hidden_size, heads, dropout):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = Sequential('x, edge_index', [
                    (GATConv( 10, hidden_size, heads, dropout), 'x, edge_index -> x'),
                    nn.ELU(inplace=True),
                    (GATConv(hidden_size*heads, hidden_size*2, heads, dropout), 'x, edge_index -> x'),
                    nn.ELU(inplace=True),
                    (GATConv(hidden_size*heads*2,hidden_size*2, heads,dropout), 'x, edge_index -> x'),
                    nn.ELU(inplace=True),
                    (GATConv(hidden_size*heads*2, hidden_size, heads=1,dropout=0.7), 'x, edge_index -> x'),
                    nn.ELU(inplace=True),
                    Linear(hidden_size, hidden_size*4),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                    Linear(hidden_size*4, hidden_size*4),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                    Linear(hidden_size*4, 1),
        ])
    def forward(self, data):
        x, edge_index = data.x.to(torch.float), data.edge_index
        x =self.layers(x, edge_index)
        return torch.sigmoid(x)


def test(model, test_data):
    model.eval()
    out = model(test_data)
    pred = out[test_data.mask].squeeze()
    index = np.where(test_data.mask)[0]
    return pd.DataFrame({"node idx":index,"node anomaly score":pred.detach().cpu().numpy()})

#test data path 
test_data = torch.load("./dataset/test_sub-graph_tensor_noLabel.pt")
test_data.mask = np.load("./dataset/test_mask.npy")
test_data.to(device)
model = GAT(2,16,1,0.7)
model = torch.load("model.pth")
model.to(device)
output = test(model,test_data)
output.to_csv("submission.csv",index =False)