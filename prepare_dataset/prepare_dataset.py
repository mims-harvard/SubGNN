# General
import numpy as np
import random
import typing
import logging
from collections import Counter, defaultdict

import config_prepare_dataset as config
import os
if not os.path.exists(config.DATASET_DIR):
    os.makedirs(config.DATASET_DIR)

import train_node_emb

# Pytorch
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# NetworkX
import networkx as nx
from networkx.generators.random_graphs import barabasi_albert_graph, extended_barabasi_albert_graph
from networkx.generators.duplication import duplication_divergence_graph


class SyntheticGraph():

    def __init__(self, base_graph_type: str, subgraph_type: str, 
                features_type: str, base_graph=None, feature_matrix=None, **kwargs):
        self.base_graph_type = base_graph_type
        self.subgraph_type = subgraph_type
        self.features_type = features_type

        self.graph = self.generate_base_graph(**kwargs)
        self.subgraphs = self.generate_and_add_subgraphs(**kwargs)
        self.subgraph_labels = self.generate_subgraph_labels(**kwargs)

        self.feature_matrix = self.initialize_features(**kwargs)


    def generate_base_graph(self, **kwargs):
        """
        Generate the base graph.

        Return
            - G (networkx object): base graph
        """

        if self.base_graph_type == 'barabasi_albert':
            m = kwargs.get('m', 5)
            n = kwargs.get('n', 500)
            G = barabasi_albert_graph(n, m, seed=config.RANDOM_SEED)
        elif self.base_graph_type == 'duplication_divergence_graph':
            n = kwargs.get('n', 500)
            p = kwargs.get('p', 0.5)
            G = duplication_divergence_graph(n, p, seed=config.RANDOM_SEED)
        else:
            raise Exception('The base graph you specified is not implemented')
        return G
    
    def initialize_features(self, **kwargs):
        """
        Initialize node features in base graph.

        Return
            - Numpy matrix
        """

        n_nodes = len(self.graph.nodes)
        if self.features_type == 'one_hot':
            return np.eye(n_nodes, dtype=int)
        elif self.features_type == 'constant':
            n_features = kwargs.pop('n_features', 20)
            return np.full((n_nodes, n_features), 1)
        else:
            raise Exception('The feature initialization you specified is not implemented')

    def generate_and_add_subgraphs(self, **kwargs):
        """
        Generate and add subgraphs to the base graph.

        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        n_subgraphs = kwargs.pop('n_subgraphs', 3)
        n_nodes_in_subgraph = kwargs.pop('n_subgraph_nodes', 5)
        n_connected_components = kwargs.pop('n_connected_components', 1)
        modify_graph_for_properties = kwargs.pop('modify_graph_for_properties', False)
        desired_property = kwargs.get('desired_property', None)

        if self.subgraph_type == 'random':
            subgraphs = self._get_subgraphs_randomly(n_subgraphs, n_nodes_in_subgraph, **kwargs)
        elif self.subgraph_type == 'bfs':
            subgraphs =  self._get_subgraphs_by_bfs(n_subgraphs, n_nodes_in_subgraph, n_connected_components, **kwargs)
        elif self.subgraph_type == 'staple':
            subgraphs = self._get_subgraphs_by_k_hops(n_subgraphs, n_nodes_in_subgraph, n_connected_components, **kwargs)
        elif self.subgraph_type == 'plant':
            if desired_property == 'coreness':
                subgraphs = self._get_subgraphs_by_coreness(n_subgraphs, n_nodes_in_subgraph, n_connected_components, **kwargs)
            else:
                subgraphs = self._get_subgraphs_by_planting(n_subgraphs, n_nodes_in_subgraph, n_connected_components, **kwargs)
        else:
            raise Exception('The subgraph generation you specified is not implemented')

        if modify_graph_for_properties:
            self._modify_graph_for_desired_subgraph_properties(subgraphs, **kwargs) 
            self._relabel_nodes(subgraphs, **kwargs) 

        return subgraphs

    def _get_subgraphs_randomly(self, n_subgraphs, n_nodes_in_subgraph, **kwargs):
        """
        Randomly generates subgraphs of size n_nodes_in_subgraph

        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph

        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        subgraphs = []
        for s in range(n_subgraphs):
            sampled_nodes = random.sample(self.graph.nodes, n_nodes_in_subgraph)
            subgraphs.append(sampled_nodes)
        return subgraphs

    def staple_component_to_graph(self, n_nodes_in_subgraph, graph_root_node, **kwargs):
        """
        Staple a connected component to a graph.

        Args
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - graph_root_node (int): node in the base graph that the component should be "stapled" to

        Return
            - cc_node_ids (list): nodes in a connected component
            - cc_root_node (int): node in the connected component (subgraph) to connect with the graph_root_node
        """
        
        # Create new connected component for the node in base graph
        con_component = self.generate_subgraph(n_nodes_in_subgraph, **kwargs)

        cc_node_ids =  list(range(len(self.graph.nodes), len(self.graph.nodes) + n_nodes_in_subgraph ))
        
        # Staple the connected component to the base graph
        joined_graph = nx.disjoint_union(self.graph, con_component)
        cc_root_node = random.sample(cc_node_ids, 1)[0]
        joined_graph.add_edge(graph_root_node, cc_root_node)
        self.graph = joined_graph.copy()

        return cc_node_ids, cc_root_node
    
    def _get_subgraphs_by_k_hops(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components, **kwargs):
        """
        Generate subgraphs that are k hops apart, staple each subgraph to the base graph by adding edge between a random node
        from the subgraph and a random node from the base graph

        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
        
        Return
            - validated_subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        diameter = nx.diameter(self.graph)
        k_hops_range = [int(diameter * k) for k in config.K_HOPS_RANGE]
        p_range = [float(p) for p in config.BA_P_RANGE]
        cc_range = [int(cc) for cc in config.CC_RANGE]
        shuffle_cc = False
        if n_connected_components == None: shuffle_cc = True
        print("DIAMETER: ", diameter)
        print("K-HOPS RANGE: ",  k_hops_range)
        print("N CONNECTED COMPONENTS: ", n_connected_components)

        subgraphs = []
        original_node_ids = self.graph.nodes

        for s in range(n_subgraphs):
            curr_subgraph = []
            seen_nodes = [] 
            all_cc_start_nodes = []
            k_hops = random.sample(k_hops_range, 1)[0]
            p = p_range[k_hops_range.index(k_hops)]
            kwargs['p'] = p

            # Randomly select a node from base graph
            graph_root_node = random.sample(original_node_ids, 1)[0]
            seen_nodes.append(graph_root_node) 
            cc_node_ids, cc_root_node = self.staple_component_to_graph(n_nodes_in_subgraph, graph_root_node, **kwargs)
            curr_subgraph.extend(cc_node_ids)
            seen_nodes.extend(cc_node_ids)
            all_cc_start_nodes.append(cc_root_node) # keep track of start nodes across CCs
            
            # Get nodes that are k hops away
            n_hops_paths = nx.single_source_shortest_path_length(self.graph, graph_root_node, cutoff=k_hops)
            candidate_nodes = [node for node in n_hops_paths if self.is_k_hops_from_all_cc(node, all_cc_start_nodes, k_hops) and node not in seen_nodes]

            if len(candidate_nodes) == 0: candidate_nodes = [node for node, length in n_hops_paths.items() if length == max(n_hops_paths.values())]
            if shuffle_cc: n_connected_components = random.sample(cc_range, 1)[0]

            for c in range(n_connected_components - 1):
                new_graph_root_node = random.sample(candidate_nodes, 1)[0] # choose a random node that is k hops away
                seen_nodes.append(new_graph_root_node) 
                cc_node_ids, cc_root_node = self.staple_component_to_graph(n_nodes_in_subgraph, new_graph_root_node, **kwargs)
                curr_subgraph.extend(cc_node_ids)
                seen_nodes.extend(cc_node_ids)
                all_cc_start_nodes.append(cc_root_node) # keep track of start nodes across CCs
            if len(curr_subgraph) >= n_nodes_in_subgraph * n_connected_components: 
                actual_num_cc = nx.number_connected_components(self.graph.subgraph(curr_subgraph))
                if shuffle_cc and actual_num_cc in config.CC_RANGE: subgraphs.append(curr_subgraph)
                elif not shuffle_cc and actual_num_cc > 1: subgraphs.append(curr_subgraph) # must have >1 CC

        # Validate that subgraphs have the desired number of CCs
        validated_subgraphs = []
        for s in subgraphs:
            actual_num_cc = nx.number_connected_components(self.graph.subgraph(s))
            if shuffle_cc and actual_num_cc in config.CC_RANGE: validated_subgraphs.append(s)
            elif not shuffle_cc and actual_num_cc > 1: validated_subgraphs.append(s) # must have >1 CC
        print(len(validated_subgraphs))
        return validated_subgraphs

    def _get_subgraphs_by_coreness(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components, remove_edges=False, **kwargs):
        """
        Sample nodes from the base graph that have at least n nodes with k core. Merge the edges from the generated
        subgraph with the edges from the base graph. Optionally, remove all other edges in the subgraphs

        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
            - remove_edges (bool): true if should remove unmerged edges in subgraphs, false otherwise
        
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        subgraphs = []

        k_core_dict = nx.core_number(self.graph)        
        nodes_per_k_core = Counter(list(k_core_dict.values()))
        print(nodes_per_k_core)
        
        nodes_with_core_number = defaultdict()
        for n, k in k_core_dict.items():
            if k in nodes_with_core_number: nodes_with_core_number[k].append(n)
            else: nodes_with_core_number[k] = [n]

        for k in nodes_with_core_number:

            # Get nodes with core number k that have not been sampled already
            nodes_with_k_cores = nodes_with_core_number[k]
            
            # Sample n_subgraphs subgraphs per core number
            for s in range(n_subgraphs):

                curr_subgraph = []
                for c in range(n_connected_components):
                    if len(nodes_with_k_cores) < n_nodes_in_subgraph: break

                    con_component = self.generate_subgraph(n_nodes_in_subgraph, **kwargs)
                    cc_node_ids = random.sample(nodes_with_k_cores, n_nodes_in_subgraph)

                    # Relabel subgraph to have the same ids as the randomly sampled nodes
                    cc_id_mapping = {curr_id:new_id for curr_id, new_id in zip(con_component.nodes, cc_node_ids)}
                    nx.relabel_nodes(con_component, cc_id_mapping, copy=False)
            
                    if remove_edges:
                        # Remove the existing edges between nodes in the planted subgraph (except the ones to be added)
                        self.graph.remove_edges_from(self.graph.subgraph(cc_node_ids).edges)

                    # Combine the base graph & subgraph. Nodes with the same ID are merged
                    joined_graph = nx.compose(self.graph, con_component) #NOTE: attributes from subgraph take precedent over attributes from self.graph
                    self.graph = joined_graph.copy()
                    
                    curr_subgraph.extend(cc_node_ids) # add nodes to subgraph
                    nodes_with_k_cores = list(set(nodes_with_k_cores).difference(set(cc_node_ids)))
                    nodes_with_core_number[k] = nodes_with_k_cores
                
                if len(curr_subgraph) > 0: subgraphs.append(curr_subgraph)

        return subgraphs

    def _get_subgraphs_by_bfs(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components,  **kwargs):
        """
        Sample n_connected_components number of start nodes from the base graph. Perform BFS to create subgraphs
        of size n_nodes_in_subgraph.

        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
        
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        max_depth = kwargs.pop('max_depth', 3)

        subgraphs = []
        for s in range(n_subgraphs):

            #randomly select start nodes. # of start nodes == n connected components
            curr_subgraph = []
            start_nodes = random.sample(self.graph.nodes, n_connected_components)            
            for start_node in start_nodes:
                edges = nx.bfs_edges(self.graph, start_node, depth_limit=max_depth)
                nodes = [start_node] + [v for u, v in edges]
                nodes = nodes[:n_nodes_in_subgraph] #limit nodes to n_nodes_in_subgraph

                if max(nodes) > max(self.graph.nodes): print(max(nodes), max(self.graph.nodes))
                assert max(nodes) <= max(self.graph.nodes)

                assert nx.is_connected(self.graph.subgraph(nodes)) #check to see if selected nodes represent a conencted component
                curr_subgraph.extend(nodes)
            subgraphs.append(curr_subgraph)
        
        seen = []
        for g in subgraphs:
            seen += g
        assert max(seen) <= max(self.graph.nodes)
        
        return subgraphs

    def generate_subgraph(self, n_nodes_in_subgraph, **kwargs):
        """
        Generate a subgraph with specified properties.

        Args
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
        
        Return
            - G (networkx object): subgraph
        """
        
        subgraph_generator = kwargs.pop('subgraph_generator', 'path')

        if subgraph_generator == 'cycle':
            G = nx.cycle_graph(n_nodes_in_subgraph)
        elif subgraph_generator == 'path':
            G = nx.path_graph(n_nodes_in_subgraph)
        elif subgraph_generator == 'house':
            G = nx.house_graph()
        elif subgraph_generator == 'complete':
            G = nx.complete_graph(n_nodes_in_subgraph)
        elif subgraph_generator == 'star':
            G = nx.star_graph(n_nodes_in_subgraph)
        elif subgraph_generator == 'barabasi_albert':
            m = kwargs.get('m', 5)
            G = barabasi_albert_graph(n_nodes_in_subgraph, m, seed=config.RANDOM_SEED)
        elif subgraph_generator == 'extended_barabasi_albert':
            m = kwargs.get('m', 5)
            p = kwargs.get('p', 0.5)
            q = kwargs.get('q', 0)
            G = extended_barabasi_albert_graph(n_nodes_in_subgraph, m, p, q, seed=config.RANDOM_SEED)
        elif subgraph_generator == 'duplication_divergence_graph':
            p = kwargs.get('p', 0.5)
            G = duplication_divergence_graph(n_nodes_in_subgraph, p)
        else:
            raise Exception('The subgraph generator you specified is not implemented.')
        return G

    def is_k_hops_away(self, start, end, n_hops):
        """
        Check whether the start node is k hops away from the end node.

        Args
            - start (int): start node
            - end (int): end node
            - n_hops (int): k hops
        
        Return
            - True if the start node is k hops away from the end node, false otherwise
        """

        shortest_path_lengh = nx.shortest_path_length(self.graph, start, end)
        if shortest_path_lengh == n_hops:
            return True
        else:
            return False

    def is_k_hops_from_all_cc(self, cand, all_cc_start_nodes, k_hops):
        """
        Check whether the candidate node is k hops away from all CC start nodes.

        Args
            - cand (int): candidate node
            - all_cc_start_nodes (list): cc start nodes
            - k_hops (int): k hops
        
        Return
            - True if the candidate node is k hops away from all CC start nodes, false otherwise
        """

        for cc_start in all_cc_start_nodes:
            if not self.is_k_hops_away(cc_start, cand, k_hops):
                return False
        return True

    def _get_subgraphs_by_stapling(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components, **kwargs):
        """
        Generate n subgraphs, staple each subgraph to the base graph by adding an edge between random node
        from the subgraph and a random node from the base graph.

        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
        
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        k_core_to_sample = kwargs.pop('k_core_to_sample', -1)
        k_hops = kwargs.pop('k_hops', -1)

        subgraphs = []
        original_node_ids = self.graph.nodes

        for s in range(n_subgraphs):
            curr_subgraph = []
            all_cc_start_nodes = []

            for c in range(n_connected_components):
                con_component = self.generate_subgraph(n_nodes_in_subgraph, **kwargs)
                graph_root_node = random.sample(original_node_ids, 1)[0]

                if c > 0 and k_hops != -1:
                    # make sure to sample the next node k hops away from the previously sampled root node
                    # and check to see that the selected start node is k hops away from all previous start nodes
                    n_hops_paths = nx.single_source_shortest_path_length(self.graph, cc_root_node, cutoff=k_hops)
                    candidate_nodes = [node for node,length in n_hops_paths.items()]
                    random.shuffle(candidate_nodes)
                    candidate_nodes = [cand for cand in candidate_nodes if self.is_k_hops_from_all_cc(cand, all_cc_start_nodes, k_hops)]
                    if len(candidate_nodes) == 0:
                        raise Exception('There are no nodes that are k hops away from all other CC start nodes.')
                    cc_root_node = random.sample(candidate_nodes, 1)[0]
                    all_cc_start_nodes.append(cc_root_node) # keep track of start nodes across CCs
                elif k_core_to_sample != -1:
                    k_core_dict = nx.core_number(self.graph)
                    nodes_with_core_number = [node for node, core_num in k_core_dict.items()if core_num == k_core_to_sample]
                    cc_root_node = random.sample(nodes_with_core_number, 1)[0]
                    all_cc_start_nodes.append(cc_root_node) # keep track of start nodes across CCs
                else: # if we're not trying to sample each CC k hops away OR if it's the first time we sample a CC, 
                      # just randomly sample a start node from the graph
                    #randomly sample root node where the CC will be attached
                    cc_node_ids =  list(range(len(self.graph.nodes), len(self.graph.nodes) + n_nodes_in_subgraph ))
                    cc_root_node = random.sample(cc_node_ids, 1)[0]
                    all_cc_start_nodes.append(cc_root_node) # keep track of start nodes across CCs

                #combine the generated subgraph & the graph
                joined_graph = nx.disjoint_union(self.graph, con_component)

                # add an edge between one node in the graph & subgraph
                joined_graph.add_edge(graph_root_node, cc_root_node)
                self.graph = joined_graph.copy()

                #add connected component to IDs for current subgraph
                curr_subgraph.extend(cc_node_ids)

            subgraphs.append(curr_subgraph)

        return subgraphs

    def _get_subgraphs_by_planting(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components, remove_edges=False, **kwargs):
        """
        Randomly sample nodes from base graph that will be in each subgraph. Merge the edges from the generated
        subgraph with the edges from the base graph. Optionally, remove all other edges in the subgraphs

        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
            - remove_edges (bool): true if should remove unmerged edges in subgraphs, false otherwise
        
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """
        
        k_core_to_sample = kwargs.pop('k_core_to_sample', -1)

        subgraphs = []
        for s in range(n_subgraphs):
            curr_subgraph = []
            for c in range(n_connected_components):

                con_component = self.generate_subgraph(n_nodes_in_subgraph, **kwargs)

                #randomly sample which nodes from the base graph will be the subgraph
                if k_core_to_sample != -1:
                    k_core_dict = nx.core_number(self.graph)
                    nodes_with_core_number = [node for node, core_num in k_core_dict.items()if core_num == k_core_to_sample]
                    cc_node_ids = random.sample(nodes_with_core_number, n_nodes_in_subgraph)
                else:
                    cc_node_ids = random.sample(self.graph.nodes, n_nodes_in_subgraph)

                #relabel subgraph to have the same ids as the randomly sampled nodes
                cc_id_mapping = {curr_id:new_id for curr_id, new_id in zip(con_component.nodes, cc_node_ids)}
                nx.relabel_nodes(con_component, cc_id_mapping, copy=False)
                
                if remove_edges:
                    #remove the existing edges between nodes in the planted subgraph (except the ones to be added)
                    self.graph.remove_edges_from(self.graph.subgraph(cc_node_ids).edges)

                # combine the base graph & subgraph. Nodes with the same ID are merged
                joined_graph = nx.compose(self.graph, con_component) #NOTE: attributes from subgraph take precedent over attributes from self.graph

                self.graph = joined_graph.copy()
                curr_subgraph.extend(cc_node_ids)
            subgraphs.append(curr_subgraph)

        return subgraphs


    def _get_property(self, subgraph, subgraph_property):
        """
        Compute the value of a specified property.

        Args
            - subgraph (networkx object): subgraph
            - subgraph_property (str): desired property of subgraph
        
        Return
            - Value of subgraph
        """

        if subgraph_property == 'density':
            return nx.density(subgraph) 

        elif subgraph_property == 'cut_ratio':
            nodes_except_subgraph = set(self.graph.nodes).difference(set(subgraph.nodes))
            n_boundary_edges = len(list(nx.edge_boundary(self.graph, subgraph.nodes, nodes_except_subgraph)))
            n_nodes = len(list(self.graph.nodes))
            n_sugraph_nodes = len(list(subgraph.nodes))
            return n_boundary_edges / (n_sugraph_nodes * (n_nodes - n_sugraph_nodes))

        elif subgraph_property == 'coreness': 
            all_cores = nx.core_number(subgraph)            
            avg_coreness = np.average(list(all_cores.values()))
            return avg_coreness

        elif subgraph_property == 'cc':
            return nx.number_connected_components(self.graph.subgraph(subgraph))

        else:
            raise Exception('The subgraph property you specificed is not implemented.')

    def _modify_graph_for_desired_subgraph_properties(self, subgraphs, **kwargs):
        """
        Modify the graph to achieve the desired subgraph property.

        Args
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """
        
        desired_property = kwargs.get('desired_property', 'density')

        # Iterate through subgraphs
        for s in subgraphs:
            subgraph = self.graph.subgraph(s)

            # DENSITY
            if desired_property == 'density':

                # Randomly select a density value
                desired_prop_value = random.sample(config.DENSITY_RANGE, 1)[0]
                
                n_tries = 0
                while True:
                    
                    curr_subg_property = self._get_property(subgraph, desired_property)
                    
                    if abs(curr_subg_property - desired_prop_value) < config.DENSITY_EPSILON: break
                    if n_tries >= config.MAX_TRIES: break

                    if curr_subg_property > desired_prop_value: #remove edges to decrease density
                        sampled_edge = random.sample(subgraph.edges, 1)[0]
                        self.graph.remove_edge(*sampled_edge)
                        
                    else: # add edges to increase density
                        sampled_nodes = random.sample(subgraph.nodes, 2)
                        self.graph.add_edge(*sampled_nodes)
                    
                    n_tries += 1
        
            # CUT RATIO
            elif desired_property == 'cut_ratio':
                
                # Randomly select a cut ratio value
                desired_prop_value = random.sample(config.CUT_RATIO_RANGE, 1)[0]

                n_tries = 0
                while True:

                    curr_subg_property = self._get_property(subgraph, desired_property)
                    
                    if abs(curr_subg_property - desired_prop_value) < config.CUT_RATIO_EPSILON: break
                    if n_tries >= config.MAX_TRIES: break

                    # get edges on boundary
                    nodes_except_subgraph = set(self.graph.nodes).difference(set(subgraph.nodes))
                    subgraph_boundary_edges = list(nx.edge_boundary(self.graph, subgraph.nodes, nodes_except_subgraph))

                    if curr_subg_property > desired_prop_value: # high cut ratio -> too many edges
                        edge_to_remove = random.sample(subgraph_boundary_edges, 1)[0]
                        self.graph.remove_edge(*edge_to_remove)
                        
                    else: # low cut ratio -> too few edges -> add edge
                        sampled_subgraph_node = random.sample(subgraph.nodes, 1)[0]
                        sampled_rest_graph_node = random.sample(nodes_except_subgraph,1)[0]
                        self.graph.add_edge(sampled_subgraph_node, sampled_rest_graph_node)

                    n_tries += 1
                        
            elif desired_property == 'coreness' or desired_property == 'cc':
                continue
            
            else:
                raise Exception('Other properties have not yet been implemented')
         
    def _relabel_nodes(self, subgraphs, **kwargs):
        """
        Relabel nodes in the graph and subgraphs to ensure that all nodes are indexed consecutively 
        """
        largest_cc = max(nx.connected_components(self.graph), key=len) 
        removed_nodes = set(list(self.graph.nodes)).difference(set(largest_cc)) 
        print("Original graph: %d, Largest cc: %d, Removed nodes: %d" % (len(self.graph.nodes), len(largest_cc), len(removed_nodes))) 
        self.graph = self.graph.subgraph(largest_cc)
        mapping = {k: v for k, v in zip(list(self.graph.nodes), range(len(self.graph.nodes)))} 
        self.graph = nx.relabel_nodes(self.graph, mapping) 
        new_subgraphs = [] 
        for s in subgraphs:
            new_s = [mapping[n] for n in s if n not in removed_nodes] 
            new_subgraphs.append(new_s) 
        return new_subgraphs 
                    
    def generate_subgraph_labels(self, **kwargs):
        """
        Generate subgraph labels

        Return
            - labels (list): subgraph labels
        """

        # Make sure base graph is connected
        if nx.is_connected(self.graph) == False:
            max_cc = max(nx.connected_components(self.graph), key=len)
            self.graph = self.graph.subgraph(max_cc)

        # Setup
        densities = []
        cut_ratios = []
        coreness = []
        cc = []
        desired_property = kwargs.get('desired_property', 'density')

        for subgraph_nodes in self.subgraphs:
            
            subgraph = self.graph.subgraph(subgraph_nodes).copy()
            
            if desired_property == 'density': 
                value = self._get_property(subgraph, desired_property)
                densities.append(value) 
            elif desired_property == 'cut_ratio': 
                value = self._get_property(subgraph, desired_property)
                cut_ratios.append(value)
            elif desired_property == 'coreness': 
                value = self._get_property(subgraph, desired_property)
                coreness.append(value)
            elif desired_property == 'cc':
                value = self._get_property(subgraph, desired_property)
                cc.append(value)

        if desired_property == 'density':  
            bins = self.generate_bins(sorted(densities), len(config.DENSITY_RANGE))
            labels = np.digitize(densities, bins = bins)
            labels = self.convert_number_to_chr(labels)
            print(Counter(labels))
            return labels

        elif desired_property == 'cut_ratio': 
            bins = self.generate_bins(sorted(cut_ratios), len(config.CUT_RATIO_RANGE))
            labels = np.digitize(cut_ratios, bins = bins)
            labels = self.convert_number_to_chr(labels)
            print(Counter(labels))
            return labels

        elif desired_property == 'coreness': 
            n_bins = kwargs.pop('n_bins', 5)
            bins = self.generate_bins(sorted(coreness), n_bins)
            labels = np.digitize(coreness, bins = bins)
            labels = self.convert_number_to_chr(labels)
            print(Counter(labels))
            return labels

        elif desired_property == 'cc':
            print(Counter(cc))
            bins = [1, 5] # 1 CC vs. >1 CC
            labels = np.digitize(cc, bins = bins)
            labels = self.convert_number_to_chr(labels)
            print(Counter(labels))
            assert len(list(Counter(labels).keys())) == len(bins)
            return labels

        else: 
            raise Exception('Other properties have not yet been implemented')

    def generate_bins(self, values, n_bins):
        """
        Generate bins for given subgraph values.

        Args
            - values (list): values for each subgraph
            - n_bins (int): number of pins to split the subgraph values into

        Return
            - bins (list): cutoffs values for each bin
        """

        bins = (len(values) / float(n_bins)) * np.arange(1, n_bins + 1)
        bins = np.unique(np.array([values[int(b) - 1] for b in bins]))
        bins = np.delete(bins, len(bins) - 1)
        print("Bins: ", bins, "Min: ", min(values), "Max: ", max(values))
        return bins

    def convert_number_to_chr(self, labels):
        """
        Convert label bins from int to str.

        Args
            - labels (list): subgraph labels

        Return
            - new_labels (list): converted subgraph labels as strings
        """

        types = {}
        alpha_int = 65 # A
        
        # Create new keys
        for t in set(labels):
            types[t] = chr(alpha_int)
            alpha_int += 1
        
        # Convert labels
        new_labels = []
        for l in labels:
            new_labels.append(types[l])
        return new_labels


def generate_mask(n_subgraphs):
    """
    Generate train/val/test masks for the subgraphs.

    Args
        - n_subgraphs (int): number of subgraphs

    Return
        - mask (list): 0 if subgraph is in train set, 1 if in val set, 2 if in test set
    """
    
    idx = set(range(n_subgraphs))
    train_mask = list(random.sample(idx, int(len(idx) * 0.8))) 
    idx = idx.difference(set(train_mask)) 
    val_mask = list(random.sample(idx, len(idx) // 2)) 
    idx = idx.difference(set(val_mask)) 
    test_mask = list(random.sample(idx, len(idx))) 
    mask = [] 
    for i in range(n_subgraphs):
        if i in train_mask: mask.append(0)  
        elif i in val_mask: mask.append(1)  
        elif i in test_mask: mask.append(2)  
    return mask


def write_f(sub_f, sub_G, sub_G_label, mask):
    """
    Write subgraph information into the appropriate format for SubGNN (tab-delimited file where each row
    has dash-delimited nodes, subgraph label, and train/val/test label).

    Args
        - sub_f (str): file directory to save subgraph information
        - sub_G (list of lists): list of subgraphs, where each subgraph is a list of nodes
        - sub_G_label (list): subgraph labels
        - mask (list): 0 if subgraph is in train set, 1 if in val set, 2 if in test set
    """

    with open(sub_f, "w") as fout:
        for g, l, m in zip(sub_G, sub_G_label, mask):
            g = [str(val) for val in g]
            if len(g) == 0: continue
            if m == 0: fout.write("\t".join(["-".join(g), str(l), "train", "\n"]))
            elif m == 1: fout.write("\t".join(["-".join(g), str(l), "val", "\n"]))
            elif m == 2: fout.write("\t".join(["-".join(g), str(l), "test", "\n"]))






def main():
    if config.GENERATE_SYNTHETIC_G: 
        synthetic_graph = SyntheticGraph(base_graph_type = config.BASE_GRAPH_TYPE,
                                         subgraph_type = config.SUBGRAPH_TYPE,
                                         n_subgraphs = config.N_SUBGRAPHS,
                                         n_connected_components = config.N_CONNECTED_COMPONENTS,
                                         n_subgraph_nodes = config.N_SUBGRAPH_NODES,
                                         features_type = config.FEATURES_TYPE,
                                         n = config.N,
                                         p = config.P,
                                         q = config.Q,
                                         m = config.M,
                                         n_bins = config.N_BINS,
                                         subgraph_generator = config.SUBGRAPH_GENERATOR,
                                         modify_graph_for_properties = config.MODIFY_GRAPH_FOR_PROPERTIES,
                                         desired_property = config.DESIRED_PROPERTY)
        nx.write_edgelist(synthetic_graph.graph, str(config.DATASET_DIR / "edge_list.txt"), data=False) 
        sub_G = synthetic_graph.subgraphs
        sub_G_label = synthetic_graph.subgraph_labels
        mask = generate_mask(len(sub_G_label)) 
        write_f(str(config.DATASET_DIR / "subgraphs.pth"), sub_G, sub_G_label, mask)
    if config.GENERATE_NODE_EMB: train_node_emb.generate_emb() 


if __name__ == "__main__":
    main()
