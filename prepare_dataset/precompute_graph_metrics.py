# General
import networkx as nx
import sys
import argparse
import snap
from pathlib import Path
import numpy as np
import json
import os
import multiprocessing

# Our methods
sys.path.insert(0, '../SubGNN') # add to path
import config

'''
Use this script to precompute information about the underlying base graph.
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Precompute graph data")
    parser.add_argument('-dataset', type=str, help='Specify the dataset folder containing the edge list for the base graph (e.g. hpo_metab')
    parser.add_argument('-calculate_shortest_paths', action='store_true', help='Calculate pairwise shortest paths between all nodes in the graph')
    parser.add_argument('-calculate_degree_sequence', action='store_true', help='Create a dictionary containing degrees of the nodes in the graph')
    parser.add_argument('-calculate_ego_graphs', action='store_true', help='Calculate the 1-hop ego graph associated with each node in the graph')
    parser.add_argument('-override', action='store_true', help='Overwrite a file even if it exists')
    parser.add_argument('-n_processes', type=int, default=4, help='Number of cores to use for multi-processsing')
    args = parser.parse_args()  
    return args

def get_shortest_path( node_id):
    NIdToDistH = snap.TIntH()
    path_len = snap.GetShortPath(snap_graph, int(node_id), NIdToDistH)
    paths = np.zeros((max(node_ids) + 1)) #previously was n_nodes
    for dest_node in NIdToDistH: 
        paths[dest_node] = NIdToDistH[dest_node]
    return paths

def calculate_stats(args):

    # create similarities folder
    if not os.path.exists(config.PROJECT_ROOT / args.dataset /'similarities'):
        os.makedirs(config.PROJECT_ROOT / args.dataset /'similarities')

    if args.calculate_ego_graphs:
        print(f'Calculating ego graphs for {args.dataset }...')
        if not (config.PROJECT_ROOT / args.dataset / 'ego_graphs.txt').exists() or override:
            ego_graph_dict = {}
            for node in snap_graph.Nodes():
                node_id = int(node.GetId())
                nodes_vec = snap.TIntV()
                snap.GetNodesAtHop(snap_graph, node_id, 1, nodes_vec, False)
                ego_graph_dict[node_id] = list(nodes_vec)
            
            with open(str(config.PROJECT_ROOT / args.dataset / 'ego_graphs.txt'), 'w') as f:
                json.dump(ego_graph_dict, f)

    if args.calculate_degree_sequence:
        print(f'Calculating degree sequences for {args.dataset }...')
        if not (config.PROJECT_ROOT / args.dataset / 'degree_sequence.txt').exists() or override:
            n_nodes = len(list(snap_graph.Nodes()))
            degrees = {}
            InDegV = snap.TIntPrV()
            snap.GetNodeInDegV(snap_graph, InDegV)
            OutDegV = snap.TIntPrV()
            snap.GetNodeOutDegV(snap_graph, OutDegV)
            for item1, item2 in zip(InDegV,OutDegV) :
                degrees[item1.GetVal1()] = item1.GetVal2()
            with open(str(config.PROJECT_ROOT / args.dataset / 'degree_sequence.txt'), 'w') as f:
                json.dump(degrees, f)

    if args.calculate_shortest_paths:
        print(f'Calculating shortest paths for {args.dataset}...')
        if not (config.PROJECT_ROOT / args.dataset /'shortest_path_matrix.npy').exists() or override:


            with multiprocessing.Pool(processes=args.n_processes) as pool:
                shortest_paths = pool.map(get_shortest_path, node_ids)

            all_shortest_paths = np.stack(shortest_paths)
            np.save(str(config.PROJECT_ROOT / args.dataset / 'shortest_path_matrix.npy'), all_shortest_paths)


# parse input args
args = parse_arguments()

# get SNAP graph for the specified dataset
snap_graph = snap.LoadEdgeList(snap.PUNGraph, str(config.PROJECT_ROOT / args.dataset / 'edge_list.txt'), 0, 1)
node_ids = np.sort([node.GetId() for node in snap_graph.Nodes()])
#n_nodes = len(list(snap_graph.Nodes()))
  
# calculate graph metrics
calculate_stats(args)
