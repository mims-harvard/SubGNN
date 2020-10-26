# Generating Synthetic Graphs

## Instructions

1. Modify the `config.py` file in the current directory (`./SubGNN/synthetic_graph`)
2. Run `python generate_synthetic_graph.py` 

## Parameters

- `DESIRED_PROPERTY` is the subgraph property to be predicted (i.e. density, cut ratio, component, coreness)
- `BASE_GRAPH_TYPE` is the type of base graph to use (i.e. barabasi albert, duplication divergence graph)
- `SUBGRAPH_TYPE` is the method used to add subgraphs (i.e. bfs, staple, plant)
- `N_SUBGRAPHS` is the number of subgraphs to create
- `N_CONNECTED_COMPONENTS` is the number of connected components per subgraph
- `N_SUBGRAPH_NODES` is the number of nodes in each subgraph
- `FEATURES_TYPE` is the node features (i.e. one-hot)
- `N` is the number of nodes in the base graph
- `P` is the probability of adding an edge between existing nodes (i.e. extended barabasi albert graph) (p + q < 1)
- `Q` is the probability of rewiring existing edges (i.e. extended barabasi albert graph) (p + q < 1)
- `M` is the number of edges to attach from a new node to existing nodes (used for barabasi albert graph)
- `N_BINS` is the number of bins for subgraph values (labels)
- `SUBGRAPH_GENERATOR` is the type of graph to use as subgraphs (i.e. complete, extended barabasi albert, duplication divergence graph)
- `MODIFY_GRAPH_FOR_PROPERTIES` is the flag for whether or not to modify graphs in order to achieve the desired property 

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
        Q = 0 # not used
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
        Q = 0 # not used
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