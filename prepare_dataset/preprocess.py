# General
import numpy as np
import random
import pickle
from collections import Counter

# Pytorch
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.utils.convert import to_networkx

# NetworkX
import networkx as nx
from networkx.relabel import convert_node_labels_to_integers, relabel_nodes
from networkx.generators.random_graphs import barabasi_albert_graph

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.insert(0, '../') # add config to path
import config_prepare_dataset as config
import utils


def read_graphs(edge_f):
    """
    Read in base graph and create a Data object for Pytorch geometric

    Args
        - edge_f (str): directory of edge list

    Return
        - all_data (Data object): Data object of base graph
    """

    nx_G = nx.read_edgelist(edge_f, nodetype = int)
    feat_mat = np.eye(len(nx_G.nodes), dtype=int)
    print("Graph density", nx.density(nx_G))
    all_data = create_dataset(nx_G, feat_mat)
    print(all_data)
    assert nx.is_connected(nx_G)
    assert len(nx_G) == all_data.x.shape[0]
    return all_data


def create_dataset(G, feat_mat, split=False):
    """
    Create Data object of the base graph for Pytorch geometric

    Args
        - G (object): NetworkX graph
        - feat_mat (tensor): feature matrix for each node

    Return
        - new_G (Data object): new Data object of base graph for Pytorch geometric 
    """

    edge_index = torch.tensor(list(G.edges)).t().contiguous() 
    x = torch.tensor(feat_mat, dtype=torch.float) # Feature matrix    
    y = torch.ones(edge_index.shape[1]) 
    num_classes = len(torch.unique(y)) 

    split_idx = np.arange(len(y))
    np.random.shuffle(split_idx)
    train_idx = split_idx[ : 8 * len(split_idx) // 10]
    val_idx = split_idx[ 8 * len(split_idx) // 10 : 9 * len(split_idx) // 10]
    test_idx = split_idx[9 * len(split_idx) // 10 : ]

    # Train set
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = 1

    # Val set
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask[val_idx] = 1

    # Test set
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask[test_idx] = 1

    new_G = Data(x = x, y = y, num_classes = num_classes, edge_index = edge_index, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask) 
    return new_G


def set_data(data, all_data, minibatch):
    """
    Create per-minibatch Data object

    Args
        - data (Data object): batched dataset
        - all_data (Data object): full dataset
        - minibatch (str): NeighborSampler

    Return
        - data (Data object): base graph as Pytorch Geometric Data object
    """

    batch_size, n_id, adjs = data
    data = Data(edge_index = adjs[0], n_id = n_id, e_id = adjs[1]) 
    data.x = all_data.x[data.n_id]
    data.train_mask = all_data.train_mask[data.e_id]
    data.val_mask = all_data.val_mask[data.e_id]
    data.y = torch.ones(len(data.e_id)) 
    return data

