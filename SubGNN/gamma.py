# General
import sys
import time
import numpy as np

# Pytorch & Networkx
import torch 
import networkx as nx

# Dynamic time warping
from fastdtw import fastdtw

# Our methods
sys.path.insert(0, '..') # add config to path
import config


###########################################
# DTW of degree sequences

def get_degree_sequence(graph, nodes, degree_dict=None, internal=True):
    '''
    Returns the ordered degree sequence of a list of nodes
    '''

    # remove badding
    nodes = nodes[nodes != config.PAD_VALUE].cpu().numpy()

    subgraph = graph.subgraph(nodes)
    internal_degree_seq = [degree for node, degree in list(subgraph.degree(nodes))]

    # for the internal structure channel, the sorted internal degree sequence is used
    if internal:
        # return the internal degree sequence
        internal_degree_seq.sort()
        return internal_degree_seq

    # for the border structure channel, the sorted external degree sequence is used
    else:

        # if we have the degree dict, use that instead of recomputing the degree of each node
        if degree_dict == None:
            graph_degree_seq =  [degree for node, degree in list(graph.degree(nodes))]
        else:
            graph_degree_seq =  [degree_dict[n-1] for n in nodes]

        external_degree_seq = [full_degree - i_degree for full_degree, i_degree in zip(graph_degree_seq, internal_degree_seq)]
        external_degree_seq.sort()
        return external_degree_seq
   
def calc_dist(a, b):
    return ((max(a,b) + 1)/(min(a,b) + 1)) - 1 

def calc_dtw( component_degree, patch_degree):
    '''
    calculate dynamic time warping between the component degree sequence and the patch degree sequence
    '''
    dist, path = fastdtw(component_degree, patch_degree, dist=calc_dist)
    return 1. / (dist + 1.)


