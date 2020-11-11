# Pytorch 
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GINConv, GCNConv 
import torch.nn.functional as F

# General	
import numpy as np	
import torch
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, conv_type, dropout):
        super(TrainNet, self).__init__()

        # GINConv
        if conv_type == "gin": 
            nn1 = nn.Sequential(nn.Linear(nfeat, nhid))
            self.conv1 = GINConv(nn1)
            nn2 = nn.Sequential(nn.Linear(nhid, nclass))
            self.conv2 = GINConv(nn2)

        # GCNConv
        if conv_type == "gcn":
            self.conv1 = GCNConv(nfeat, nhid)
            self.conv2 = GCNConv(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p = self.dropout, training = self.training)
        return self.conv2(x, edge_index)
