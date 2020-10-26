# Generating Synthetic Graphs

## Instructions

1. Modify the `config.py` file in the current directory (`./SubGNN/synthetic_graph`)
2. Run `python generate_synthetic_graph.py` 

## Parameters

The following examples are the parameters used to create the synthetic graphs described in the SubGNN paper.

### Density

        DESIRED_PROPERTY = 'density'
        BASE_GRAPH_TYPE = 'barabasi_albert'
        SUBGRAPH_TYPE = 'bfs'
        N_SUBGRAPHS = 250
        N_CONNECTED_COMPONENTS = 1
        N_SUBGRAPH_NODES = 20
        FEATURES_TYPE = 'one_hot'
        N = 5000
        P = 0.5 # default
        Q = 0 # not used (Q = 1 - P)
        M = 5 # default
        N_BINS = 3 # not used
        SUBGRAPH_GENERATOR = 'complete'
        MODIFY_GRAPH_FOR_PROPERTIES = True 

### Cut Ratio

        DESIRED_PROPERTY = 'cut_ratio'
        BASE_GRAPH_TYPE = 'barabasi_albert'
        SUBGRAPH_TYPE='plant'
        N_SUBGRAPHS =  250
        N_CONNECTED_COMPONENTS = 1
        N_SUBGRAPH_NODES = 20
        FEATURES_TYPE='one_hot'
        N = 5000
        P = 0.5 # default
        Q = 0 # not used (Q = 1 - P)
        M = 5 # default
        N_BINS = 3 # not used
        SUBGRAPH_GENERATOR = 'complete' 
        MODIFY_GRAPH_FOR_PROPERTIES = True

### Component

        DESIRED_PROPERTY = 'cc'
        BASE_GRAPH_TYPE='barabasi_albert' 
        SUBGRAPH_TYPE='staple'
        N_SUBGRAPHS =  250
        N_CONNECTED_COMPONENTS = None
        N_SUBGRAPH_NODES = 15
        FEATURES_TYPE='one_hot'
        N = 1000
        P = 0.5 # default
        Q = 0
        M = 5 # default
        N_BINS = 2 # not used
        SUBGRAPH_GENERATOR = 'extended_barabasi_albert' 
        MODIFY_GRAPH_FOR_PROPERTIES = True

### Coreness

        DESIRED_PROPERTY = 'coreness'
        BASE_GRAPH_TYPE = 'duplication_divergence_graph' 
        SUBGRAPH_TYPE = 'plant'
        N_SUBGRAPHS = 30
        N_CONNECTED_COMPONENTS = 1
        N_SUBGRAPH_NODES = 20
        FEATURES_TYPE='one_hot'
        N = 5000
        P = 0.7
        Q = 0 # not used (Q = 1 - P)
        M = 1
        N_BINS = 3
        SUBGRAPH_GENERATOR = 'duplication_divergence_graph' 
        MODIFY_GRAPH_FOR_PROPERTIES = True
