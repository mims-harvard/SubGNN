# General
import numpy as np
import random
from collections import defaultdict
import networkx as nx
import sys
import time

# Pytorch
import torch

# Our Methods
sys.path.insert(0, '..') # add config to path
import config
import subgraph_utils

#######################################################
# Triangular Random Walks

def is_triangle(graph, a, b, c):
    '''
    Returns true if the nodes a,b,c consistute a triangle in the graph
    '''
    return c in set(graph.neighbors(a)).intersection(set(graph.neighbors(b)))

def get_neighbors(networkx_graph, subgraph, all_valid_border_nodes, prev_node, curr_node, inside):
    '''
    Returns lists of triangle and non-triangle neighbors for the curr_node
    '''

    # if 'inside', we don't want to consider any nodes outside of the subgraph
    if inside: graph = subgraph
    else: graph = networkx_graph

    neighbors = list(graph.neighbors(curr_node))

    # if we're doing a border random walk, we need to make sure the neighbors are in the valid set of nodes to consider (i.e. all_valid_border_nodes)
    if not inside: neighbors = [n for n in neighbors if n in all_valid_border_nodes]

    # separate neighbors into triangle and non-triangle neighbors
    triangular_neighbors = []
    non_triangular_neighbors = []
    for n in neighbors:
        if is_triangle(graph, prev_node, curr_node, n): triangular_neighbors.append(n)
        else: non_triangular_neighbors.append(n)
    
    return triangular_neighbors, non_triangular_neighbors

def triangular_random_walk(hparams, networkx_graph, anchor_patch_subgraph, walk_len, in_border_nodes, all_valid_nodes, inside):
    '''
    Perform a triangular random walk 
    This is used (1) to sample anchor patches and (2) to generate internal/border representations of the anchor patches

    when using function for (1), in_border_nodes, non_subgraph_nodes, all_valid_nodes = None; inside = True & anchor_patch_subgraph is actually the entire networkx graph

    Inputs:
        - hparams: dict of hyperparameters
        - networkx graph: base underlying graph
        - anchor patch subgraph: this is either the anchor patch subgraph or in the case of (1), it's actually the base underlying graph
        - walk_len: length of the random walk
        - in_border_nodes: nodes that are in the subgraph, but that have an edge to a node external to the subgraph
        - all_valid_nodes: the union of in_border_nodes + nodes that are not in the subgraph but are in the underlying graph
        - inside: whether this random walk is internal or border to the subgraph (note that when using this method to sample anchor patches, inside=True)

    Output:
        - visited: list of node ids visited during the walk
    '''
    if inside:
        # randomly sample a start node from the subgraph/graph
        prev_node = np.random.choice(list(anchor_patch_subgraph.nodes()))
        # get all of the neighbors for the start node
        neighbor_nodes = list(anchor_patch_subgraph.neighbors(prev_node))
        # sample node from neighbors
        curr_node = np.random.choice(neighbor_nodes) if len(neighbor_nodes) > 0 else config.PAD_VALUE #we're using the PAD_VALUE as a sentinel that the first node has no neighbors
        all_valid_nodes = None
    else:
        # ranomly sample a start node from the list of 'in_border_nodes" and restrict neighboring nodes to only those in 'all_valid_nodes'
        prev_node = np.random.choice(in_border_nodes, 1)[0]
        neighbor_nodes = [n for n in list(networkx_graph.neighbors(prev_node)) if n in all_valid_nodes]
        curr_node = np.random.choice(neighbor_nodes, 1)[0] if len(neighbor_nodes) > 0 else config.PAD_VALUE
    
    # if the first node has no neighbors, the random walk is only length 1 & we return it immediately
    if curr_node == config.PAD_VALUE:
        return [prev_node]
    
    visited = [prev_node, curr_node]
    # now that we've already performed a walk of length 2, let's perform the rest of the walk
    for k in range(walk_len - 2):

        #get the triangular and non-triangular neighbors for the current node given the previously visited node
        triangular_neighbors, non_triangular_neighbors = get_neighbors(networkx_graph, anchor_patch_subgraph, all_valid_nodes, prev_node, curr_node, inside=inside)
        neighbors = triangular_neighbors + non_triangular_neighbors

        if len(neighbors) == 0: break # if there are no neighbors, end walk
        else:
            # if there are no neighbors of one type, sample from the other type
            if len(triangular_neighbors) == 0: 
                next_node = np.random.choice(non_triangular_neighbors)
            elif len(non_triangular_neighbors) == 0: 
                next_node = np.random.choice(triangular_neighbors)
            # with probability 'rw_beta', we go to a triangular node
            elif random.uniform(0, 1) <= hparams['rw_beta'] and len(triangular_neighbors) != 0:
                next_node = np.random.choice(triangular_neighbors)
            # otherwise we go to a non-triangular node
            else:
                next_node = np.random.choice(non_triangular_neighbors)

        prev_node = curr_node
        curr_node = next_node
        visited.append(next_node)
    
    # we return a list of the node ids visited during the walk
    return visited        

#######################################################
# Perform random walks over the sampled structure anchor patches

def perform_random_walks(hparams, networkx_graph, anchor_patch_ids, inside):
    '''
    Performs random walks over the sampled anchor patches

    If inside=True, performs random walks over the inside of the subgraph. Otherwise, performs random walks over the subgraph border 
    (i.e. nodes in the subgraph that have an external edge + nodes external to the subgraph)

    Returns padded tensor of all walks of shape (n sampled anchor patches, n_triangular_walks, random_walk_len)
    '''
    n_sampled_patches, max_patch_len = anchor_patch_ids.shape

    all_patch_walks = []
    for anchor_patch in anchor_patch_ids:
        curr_anchor_patch = anchor_patch[anchor_patch != config.PAD_VALUE] #remove any padding

        # if anchor patch is only padding, then we just add a tensor of all zeros to maintain the padding
        if curr_anchor_patch.shape[0] == 0:
            all_patch_walks.append(torch.zeros((hparams['n_triangular_walks'], hparams['random_walk_len']), dtype=torch.long).fill_(config.PAD_VALUE))
        else:
            
            anchor_patch_subgraph = networkx_graph.subgraph(curr_anchor_patch.numpy()) # create a networkx graph from the anchor patch
            if not inside: 
                # get nodes in subgraph that have an edge to a node not in the subgraph & all of the nodes that are not in the subgraph
                in_border_nodes, non_subgraph_nodes = subgraph_utils.get_border_nodes(networkx_graph, anchor_patch_subgraph)
                # the border random walk can operate over all nodes on the border of the subgraph + all nodes external to the subgraph
                all_valid_nodes = set(in_border_nodes).union(set(non_subgraph_nodes))
            else: in_border_nodes, non_subgraph_nodes, all_valid_nodes = None, None, None
            
            # perform 'n_triangular_walks' number of walks over the anchor patch (each walk's length = 'random_walk_len')
            # pad the walks and stack them to produce a final tensor of shape (n sampled anchor patches, n_triangular_walks, random_walk_len)
            walks = []
            for w in range(hparams['n_triangular_walks']):
                walk = triangular_random_walk(hparams, networkx_graph, anchor_patch_subgraph, hparams['random_walk_len'], in_border_nodes, all_valid_nodes, inside=inside)
                fill_len = hparams['random_walk_len'] - len(walk)
                walk = torch.cat([torch.LongTensor(walk),torch.LongTensor((fill_len)).fill_(config.PAD_VALUE)])
                walks.append(walk)
            
            all_patch_walks.append(torch.stack(walks))
    all_patch_walks = torch.stack(all_patch_walks).view(n_sampled_patches, hparams['n_triangular_walks'], hparams['random_walk_len'])
    
    return all_patch_walks

#######################################################
# Sample anchor patches

def sample_neighborhood_anchor_patch(hparams, networkx_graph, cc_ids, border_set, sample_inside=True ):
    '''
    Returns a tensor of shape (batch_sz, max_n_cc, n_anchor_patches_N_in OR n_anchor_patches_N_out) that contains the sampled 
    neighborhood internal or border anchor patches
    '''
    batch_sz, max_n_cc, _ = cc_ids.shape
    components = cc_ids.view(cc_ids.shape[0]*cc_ids.shape[1], -1) #(batch_sz * max_n_cc, max_cc_len)

    # sample internal N anchor patch 
    if sample_inside:
        all_samples = []
        for i in range(hparams['n_anchor_patches_N_in']):
            # to efficiently sample a random element from each connected component (with variable lengths), 
            # we generate and pad a random matrix then take the argmax. This gives a randomly sampled node ID from within the component.
            rand = torch.randn(components.shape) 
            rand[components == config.PAD_VALUE] = config.PAD_VALUE
            sample = components[range(len(components)), torch.argmax(rand, dim=1)]
            all_samples.append(sample) 
        samples = torch.transpose(torch.stack(all_samples), 0, 1)

    # sample border N anchor patch 
    else:
        border_set_reshaped = border_set.view(border_set.shape[0]*border_set.shape[1], -1)
        all_samples = []
        for i in range(hparams['n_anchor_patches_N_out']): # number of neighborhood border AP to sample
            # same approach as internally, except that we're sampling from the border_set instead of within the connected component
            rand = torch.randn(border_set_reshaped.shape) 
            rand[border_set_reshaped == config.PAD_VALUE] = config.PAD_VALUE
            sample = border_set_reshaped[range(len(border_set_reshaped)), torch.argmax(rand, dim=1)]

            all_samples.append(sample)
        samples = torch.transpose(torch.stack(all_samples),0,1)

    # Reshape and return
    anchor_patches = samples.view(batch_sz, max_n_cc, -1)
    return anchor_patches

def sample_position_anchor_patches(hparams, networkx_graph, subgraph = None):
    '''
    Returns list of sampled position anchor patches. If subgraph != None, we sample from within the entire subgraph (across all CC). 
    Otherwise, we sample from the entire base graph. 'n_anchor_patches_pos_out' and 'n_anchor_patches_pos_in' specify the number of anchor patches to sample.
    '''
    if not subgraph: #sample border position anchor patches
        return list(np.random.choice(list(networkx_graph.nodes), hparams['n_anchor_patches_pos_out'], replace = True))
    else: #sampling internal position anchor patches
        return list(np.random.choice(subgraph, hparams['n_anchor_patches_pos_in'], replace = True))

def sample_structure_anchor_patches(hparams, networkx_graph, device, max_sim_epochs): 
    '''
    Generate a large number of structure anchor patches from which we can sample later

    max_sim_epochs: multiplication factor to ensure we generate more AP than are actually needed

    Returns a tensor of shape (n sampled patches, max patch length)
    '''

    # number of anchor patches to sample
    n_samples = max_sim_epochs * hparams['n_anchor_patches_structure'] * hparams['n_layers']
    all_patches = []
    start_nodes = list(np.random.choice(list(networkx_graph.nodes), n_samples, replace = True))
    for i, node in enumerate(start_nodes):

        # there are two approaches implemented to sample the structure anchor patches: 'ego_graph' or 'triangular_random_walk' (the default)
        if hparams['structure_patch_type'] == 'ego_graph': 
            # in this case, the anchor patch is the ego graph around the randomly sampled start node where the radius is specified by 'structure_anchor_patch_radius'
            subgraph = list(nx.ego_graph(networkx_graph, node, radius=hparams['structure_anchor_patch_radius']).nodes)
        elif hparams['structure_patch_type'] == 'triangular_random_walk':
            # in this case, we perform a triangular random walk of length 'sample_walk_len'
            subgraph = triangular_random_walk(hparams, networkx_graph, networkx_graph, hparams['sample_walk_len'], None,  None, True)
        else:
            raise NotImplementedError
        all_patches.append(subgraph)

    # pad the sampled anchor patches to the max length
    max_anchor_len = max([len(s) for s in all_patches])
    padded_all_patches = []
    for s in all_patches:
        fill_len = max_anchor_len - len(s)
        padded_all_patches.append(torch.cat([torch.LongTensor(s),torch.LongTensor((fill_len)).fill_(config.PAD_VALUE)]))

    return torch.stack(padded_all_patches).long() # (n sampled patches, max patch length)

#######################################################
# Initialize anchor patches

def init_anchors_neighborhood(split, hparams, networkx_graph, device, train_cc_ids, val_cc_ids, test_cc_ids, train_N_border, val_N_border, test_N_border):
    '''
    Returns:
        - anchors_int_neigh: dict of dicts mapping from dataset name & layer number -> sampled internal N anchor patches
        - anchors_border_neigh: same as above, but stores border N anchor patches
    '''

    # get datasets to process based on split
    if split == 'all':
        dataset_names = ['train', 'val', 'test']
        datasets = [train_cc_ids, val_cc_ids, test_cc_ids]
        border_sets = [train_N_border, val_N_border, test_N_border]
    elif split == 'train_val':
        dataset_names = ['train', 'val']
        datasets = [train_cc_ids, val_cc_ids]
        border_sets = [train_N_border, val_N_border]
    elif split == 'test':
        dataset_names = ['test']
        datasets = [test_cc_ids]
        border_sets = [test_N_border]

    #initialize internal and border neighborhood anchor patch dicts
    anchors_int_neigh = defaultdict(dict)
    anchors_border_neigh = defaultdict(dict)

    # for each dataset, for each layer, sample internal and border neighborhood anchor patches
    # we can use the precomputed border set to speed up the calculation
    for dataset_name, dataset, border_set in zip(dataset_names, datasets, border_sets):
        for n in range(hparams['n_layers']):
            anchors_int_neigh[dataset_name][n] = sample_neighborhood_anchor_patch(hparams, networkx_graph, dataset, border_set, sample_inside=True)
            anchors_border_neigh[dataset_name][n] = sample_neighborhood_anchor_patch(hparams, networkx_graph, dataset, border_set, sample_inside=False)
    return anchors_int_neigh, anchors_border_neigh

def init_anchors_pos_int(split, hparams, networkx_graph, device, train_cc_ids, val_cc_ids, test_cc_ids):
    '''
    Returns:
        -  anchors_pos_int: dict of dicts mapping from dataset name (e.g train, val, etc.) and layer number to the sampled internal position anchor patches
    '''
    
    # get datasets to process based on split
    if split == 'all':
        dataset_names = ['train', 'val', 'test']
        datasets = [train_cc_ids, val_cc_ids, test_cc_ids]
    elif split == 'train_val':
        dataset_names = ['train', 'val']
        datasets = [train_cc_ids, val_cc_ids]
    elif split == 'test':
        dataset_names = ['test']
        datasets = [test_cc_ids]
    
    anchors_pos_int = defaultdict(dict)
    # for each dataset, for each layer, sample internal position anchor patches
    for dataset_name, dataset in zip(dataset_names, datasets):
        for n in range(hparams['n_layers']):
            anchors = [sample_position_anchor_patches(hparams, networkx_graph, sg) for sg in dataset]
            anchors_pos_int[dataset_name][n] = torch.stack([torch.tensor(l) for l in anchors])
    return anchors_pos_int

def init_anchors_pos_ext(hparams, networkx_graph, device):
    '''
    Returns:
        - anchors_pos_ext: dict mapping from layer number in SubGNN -> tensor of sampled border position anchor patches
    '''
    anchors_pos_ext = {}
    for n in range(hparams['n_layers']):
        anchors_pos_ext[n] = torch.tensor(sample_position_anchor_patches(hparams, networkx_graph))
    return anchors_pos_ext

def init_anchors_structure(hparams, structure_anchors, int_structure_anchor_rw, bor_structure_anchor_rw):
    '''
    For each layer in SubGNN, sample 'n_anchor_patches_structure' number of anchor patches and their associated pre-computed internal & border random walks

    Returns:
        - anchors_struc: dictionary from layer number -> tuple(sampled structure anchor patches, indices of the selected anchor patches in larger list of sampled anchor patches, 
        associated sampled internal random walks, associated sampled border random walks)
    '''
    anchors_struc = {}
    for n in range(hparams['n_layers']):
        indices = list(np.random.choice(range(structure_anchors.shape[0]), hparams['n_anchor_patches_structure'], replace = True))
        anchors_struc[n] = (structure_anchors[indices,:], indices, int_structure_anchor_rw[indices,:,:], bor_structure_anchor_rw[indices,:,:] )
    return anchors_struc

#######################################################
# Retrieve anchor patches

def get_anchor_patches(dataset_type, hparams, networkx_graph, node_matrix, \
    subgraph_idx, cc_ids, cc_embed_mask, lstm, anchors_neigh_int, anchors_neigh_border, \
    anchors_pos_int, anchors_pos_ext, anchors_structure, layer_num, channel, inside, \
    device=None):
    '''
    Inputs:
        - dataset_type: train, val, etc.
        - hparams: dictionary of hyperparameters
        - networkx_graph: 
        - node_matrix: matrix containing node embeddings for every node in base graph


    Returns:
        - anchor_patches: tensor of shape (batch_sz, max_n_cc, n_anchor_patches, max_length_anchor_patch) containing the node ids associated with each anchor patch
        - anchor_mask: tensor of shape (batch_sz, max_n_cc, n_anchor_patches, max_length_anchor_patch) containing a mask over the anchor patches so we know which are just padding
        - anchor_embeds: tensor of shape (batch_sz, max_n_cc, n_anchor_patches, embed_dim) containing embeddings for each anchor patch
    '''
    batch_sz, max_n_cc, max_size_cc = cc_ids.shape

    if channel == 'neighborhood':

        # look up precomputed anchor patches
        if inside:
            anchor_patches = anchors_neigh_int[dataset_type][layer_num][subgraph_idx].squeeze(1)
        else:
            anchor_patches = anchors_neigh_border[dataset_type][layer_num][subgraph_idx].squeeze(1)
        anchor_patches = anchor_patches.to(cc_ids.device)
        
        # Get anchor patch embeddings: return shape is (batch_sz, max_n_cc, n_sampled_patches, hidden_dim)
        anchor_embeds, anchor_mask = embed_anchor_patch(node_matrix, anchor_patches, device)  
        anchor_patches = anchor_patches.unsqueeze(-1)
        anchor_mask = anchor_mask.unsqueeze(-1)

    elif channel == 'position':
        # Get precomputed anchor patch ids: return shape is (batch_sz, max_n_cc, n_sampled_patches)
        if inside:
            anchors_tensor = anchors_pos_int[dataset_type][layer_num][subgraph_idx].squeeze(1)
            anchor_patches = anchors_tensor.unsqueeze(1).repeat(1,max_n_cc,1) # repeat anchor patches for each CC
            anchor_patches[~cc_embed_mask] = config.PAD_VALUE #mask CC that are just padding
        else:
            anchor_patches = anchors_pos_ext[layer_num].unsqueeze(0).unsqueeze(0).repeat(batch_sz,max_n_cc,1)
            anchor_patches[~cc_embed_mask] = config.PAD_VALUE #mask CC that are just padding
        
        # Get anchor patch embeddings: return shape is (batch_sz, max_n_cc, n_sampled_patches, hidden_dim)
        anchor_embeds, anchor_mask = embed_anchor_patch(node_matrix, anchor_patches, device)  
        anchor_patches = anchor_patches.unsqueeze(-1)
        anchor_mask = anchor_mask.unsqueeze(-1)

    elif channel == 'structure':
        anchor_patches, indices, int_anchor_rw, bor_anchor_rw = anchors_structure[layer_num] #(n_anchor_patches_sampled, max_length_anchor_patch)

        # Get anchor patch embeddings: return shape is (n_sampled_patches, hidden_dim)
        anchor_rw = int_anchor_rw if inside else bor_anchor_rw
        anchor_embeds = aggregate_structure_anchor_patch(hparams, networkx_graph, lstm, node_matrix, anchor_patches, anchor_rw, inside=inside, device=cc_ids.device)
     
        # expand anchor patches/embeddings to be batch_sz, max_n_cc and pad them
        # return shape of anchor_patches = (bs, n_cc, n_anchor_patches_sampled, max_length_anchor_patch)
        anchor_patches = anchor_patches.unsqueeze(0).unsqueeze(0).repeat(batch_sz,max_n_cc,1,1)
        anchor_patches[~cc_embed_mask] = config.PAD_VALUE # mask CC that are just padding
        anchor_mask = (anchor_patches != config.PAD_VALUE).bool()
        anchor_embeds = anchor_embeds.unsqueeze(0).unsqueeze(0).repeat(batch_sz,max_n_cc,1,1)
        anchor_embeds[~cc_embed_mask] = config.PAD_VALUE
    else:
        raise Exception('An invalid channel has been entered.')
    
    
    return anchor_patches, anchor_mask, anchor_embeds

#######################################################
# Embed anchor patches
      
def embed_anchor_patch(node_matrix, anchor_patch_ids, device):
    '''
    Returns a tensor of the node embeddings associated with the `anchor patch ids` 
    and an associated mask where there's 1 where there's no padding and 0 otherwise
    '''
    anchor_patch_embeds = node_matrix(anchor_patch_ids.to(device))
    anchor_patch_mask = (anchor_patch_ids != config.PAD_VALUE).bool()
    return anchor_patch_embeds, anchor_patch_mask

def aggregate_structure_anchor_patch(hparams, networkx_graph, lstm, node_matrix, anchor_patch_ids, all_patch_walks, inside, device):
    '''
    Computes embedding for structure anchor patch by (1) retrieving node embeddings for nodes visited in precomputed triangular random walks, 
    (2) feeding the RW embeddings into an bi-lstm, and (3) summing the resulting embedding for each random walk to generate a 
    final embedding of shape (n sampled anchor batches, node_embed_dim)
    '''
    # anchor_patch_ids shape is (batch_sz, max_n_cc, n_sampled_patches, max_patch_len)
    # anchor_patch_embeds shape is (batch_sz, max_n_cc, n_sampled_patches, max_patch_len, hidden_dim)
    
    n_sampled_patches, max_patch_len = anchor_patch_ids.shape

    #Get embeddings for each walk
    walk_embeds, _ = embed_anchor_patch(node_matrix, all_patch_walks, device) #  n_patch, n_walk, walk_len, embed_sz
    walk_embeds_reshaped = walk_embeds.view(n_sampled_patches * hparams['n_triangular_walks'], hparams['random_walk_len'], hparams['node_embed_size'])
    
    # input into RNN & aggregate over walk len
    walk_hidden = lstm(walk_embeds_reshaped)
    walk_hidden = walk_hidden.view(n_sampled_patches, hparams['n_triangular_walks'], -1)

    # Sum over random walks
    return torch.sum(walk_hidden, dim=1) 
  


