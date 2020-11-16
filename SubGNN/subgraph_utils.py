# General
import typing
import sys
import numpy as np

#Networkx
import networkx as nx

# Sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score

# Pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import one_hot


# Our methods
sys.path.insert(0, '..') # add config to path
import config

def read_subgraphs(sub_f, split = True):
    '''
    Read subgraphs from file
    
    Args
       - sub_f (str): filename where subgraphs are stored
    
    Return for each train, val, test split:
       - sub_G (list): list of nodes belonging to each subgraph	
       - sub_G_label (list): labels for each subgraph
    '''
    
    # Enumerate/track labels
    label_idx = 0
    labels = {}


    # Train/Val/Test subgraphs
    train_sub_G = []
    val_sub_G = []
    test_sub_G = []

    # Train/Val/Test subgraph labels
    train_sub_G_label = []
    val_sub_G_label = []
    test_sub_G_label = []

    # Train/Val/Test masks
    train_mask = []
    val_mask = []
    test_mask = []

    multilabel = False

    # Parse data
    with open(sub_f) as fin:
        subgraph_idx = 0
        for line in fin:
            nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
            if len(nodes) != 0:
                if len(nodes) == 1: print(nodes)
                l = line.split("\t")[1].split("-")
                if len(l) > 1: multilabel = True
                for lab in l:    
                    if lab not in labels.keys(): 
                        labels[lab] = label_idx
                        label_idx += 1
                if line.split("\t")[2].strip() == "train":
                    train_sub_G.append(nodes)
                    train_sub_G_label.append([labels[lab] for lab in l])
                    train_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "val":
                    val_sub_G.append(nodes)
                    val_sub_G_label.append([labels[lab] for lab in l])
                    val_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "test":
                    test_sub_G.append(nodes)
                    test_sub_G_label.append([labels[lab] for lab in l])
                    test_mask.append(subgraph_idx)
                subgraph_idx += 1
    if not multilabel:
        train_sub_G_label = torch.tensor(train_sub_G_label).long().squeeze()
        val_sub_G_label = torch.tensor(val_sub_G_label).long().squeeze()
        test_sub_G_label = torch.tensor(test_sub_G_label).long().squeeze()

    if len(val_mask) < len(test_mask):
        return train_sub_G, train_sub_G_label, test_sub_G, test_sub_G_label, val_sub_G, val_sub_G_label

    return train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label

def calc_f1(logits, labels, avg_type='macro',  multilabel_binarizer=None):
    '''
    Calculates the F1 score (either macro or micro as defined by 'avg_type') for the specified logits and labelss
    '''
    if multilabel_binarizer is not None: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type)
        
    else: # multi-class, but not multi-label prediction

        pred = torch.argmax(logits, dim=-1) #get predictions by finding the indices with max logits
        score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type)
    return torch.tensor([score])

def calc_accuracy(logits, labels,  multilabel_binarizer=None):
    '''
    Calculates the accuracy for the specified logits and labels
    '''
    if multilabel_binarizer is not None: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    else:
        pred = torch.argmax(logits, 1) #get predictions by finding the indices with max logits
        acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    return torch.tensor([acc])

def get_border_nodes(graph, subgraph):
    '''
    Returns (1) an array containing the border nodes of the subgraph (i.e. all nodes that have an edge to a node not in the subgraph, but are themselves in the subgraph)
    and (2) an array containing all of the nodes in the base graph that aren't in the subgraph
    '''

    # get all of the nodes in the base graph that are not in the subgraph
    non_subgraph_nodes = np.array(list(set(graph.nodes()).difference(set(subgraph.nodes()))))

    subgraph_nodes = np.array(list(subgraph.nodes()))
    A = nx.adjacency_matrix(graph).todense()

    # subset adjacency matrix to get edges between subgraph and non-subgraph nodes
    border_A = A[np.ix_(subgraph_nodes - 1,non_subgraph_nodes - 1)] # NOTE: Need to subtract 1 bc nodes are indexed starting at 1

    # the nodes in the subgraph are border nodes if they have at least one edge to a node that is not in the subgraph
    border_edge_exists = (np.sum(border_A, axis=1) > 0).flatten()
    border_nodes = subgraph_nodes[np.newaxis][border_edge_exists]
    return border_nodes, non_subgraph_nodes

def get_component_border_neighborhood_set(networkx_graph, component, k, ego_graph_dict=None):
    '''
    Returns a set containing the nodes in the k-hop border of the specified component

    component: 1D tensor of node IDs in the component (with possible padding)
    k: number of hops around the component that is included in the border set
    ego_graph_dict: dictionary mapping from node id to precomputed ego_graph for the node
    '''

    # First, remove any padding that exists in the component
    if type(component) is torch.Tensor: 
        component_inds_non_neg = (component!=config.PAD_VALUE).nonzero().view(-1)
        component_set = {int(n) for n in component[component_inds_non_neg]}
    else:
        component_set = set(component)

    # calculate the ego graph for each node in the connected component & take the union of all nodes
    neighborhood = set()
    for node in component_set: 
        if ego_graph_dict == None: # if it hasn't already been computed, calculate the ego graph (i.e. induced subgraph of neighbors centered at node with specified radius)
            ego_g = nx.ego_graph(networkx_graph, node, radius = k).nodes()
        else:
            ego_g = ego_graph_dict[node-1] #NOTE: nodes in dict were indexed with 0, while our nodes are indexed starting at 1

        neighborhood = neighborhood.union(set(ego_g))

    # remove from the unioned ego sets all nodes that are actually in the component
    # this will leave only the nodes that are in the k-hop border, but not in the subgraph component
    border_nodes = neighborhood.difference(component_set)

    return border_nodes

# THE BELOW FUNCTIONS ARE COPIED FROM ALLEN NLP
def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.
    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
    For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
    embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:
        - `(batch_size, num_queries, num_words)` (distribution over words for each query)
        - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
          query for each document)
    are valid input "vectors", producing tensors of shape:
    `(batch_size, num_queries, embedding_dim)` and
    `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)

def masked_sum(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    **
    Adapted from AllenNLP's masked mean: 
    https://github.com/allenai/allennlp/blob/90e98e56c46bc466d4ad7712bab93566afe5d1d0/allennlp/nn/util.py
    ** 
    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    
    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    return value_sum 
