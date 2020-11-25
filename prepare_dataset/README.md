# Prepare Data

## Dataset Options

There are three options for preparing the data for SubGNN. Before doing any of these, make sure you set the `PROJECT_ROOT` in `config.py`.

### Real-World Datasets

Follow the instructions in the dataset section [here](https://github.com/mims-harvard/SubGNN#prepare-data) to download the 4 real-world datasets we provide. We have already generated the node embeddings & precomputed similarities (pairwise shortest paths, dict of node degrees, etc.)

### Synthetic Datasets

To generate synthetic datasets, first run  `python prepare_dataset.py` to generate the synthetic base graph, subgraphs, and node embeddings. To do this, make sure that the `GENERATE_SYNTHETIC_G` and `GENERATE_NODE_EMB` flags are set to `True` in the `config_prepare_dataset.py` file. Set `DATASET_DIR` to the dataset name (e.g. `density`). Then, run `python precompute_graph_metrics.py` to precompute node degrees, pairwise shortest paths, and the ego graphs for each node in the base graph. These precomputations will speed up similarity calculations in SubGNN. Afterwards, all of the necessary files to run SubGNN should exist in the `DATASET_DIR` folder. 

### Your own Data

To use your own data with SubGNN, you will need (1) an edge list for your base graph named `edge_list.txt` and (2) a file containing your subgraphs & labels named `subgraphs.pth`. The edge list is a space separated file where each line contains the start and end node id of an edge. If your graph is in networkx, you can generate this file by running [`nx.write_edgelist(GRAPH, DATASET_DIR / "edge_list.txt", data=False)`](https://networkx.org/documentation/stable//reference/readwrite/generated/networkx.readwrite.edgelist.write_edgelist.html). The subgraph file is a tab-separated file where each line contains information about a subgraph via the format `"{SUBGRAPH_IDS}\t{LABEL}\t{DATASET}\n"`. `SUBGRAPH_IDS` contains the node ids of all nodes in the subgraph joined by "-" (i.e. `"-".join(subgraph_ids)`), `LABEL` is the subgraph label as a string, and `DATASET` is either "train", "val", or "test". You can look at the 4 real-world datasets that we provide as examples of this format. 

After you have formatted your data, run `python prepare_dataset.py` to generate node embeddings (set `GENERATE_SYNTHETIC_G` to FALSE and `GENERATE_NODE_EMB` to TRUE). Set `DATASET_DIR` to the dataset name (e.g. `your_custom_data`). Then, run `python precompute_graph_metrics.py` to precompute node degrees, pairwise shortest paths, and the ego graphs for each node in the base graph. Afterwards, all of the necessary files to run SubGNN should exist in the `DATASET_DIR` folder. 

## How to Prepare Data

1. Modify `PROJECT_ROOT` in `config.py`
2. Modify `config_prepare_dataset.py` in the current directory.
3. Run `python prepare_dataset.py` 
4. Run `python precompute_graph_metrics.py`

### Output 
- `DATASET_DIR` is the directory for output files (Ex: `density`). We recommend naming the output directory as the name of your dataset. 

### Flags
- `GENERATE_SYNTHETIC_G` enables generating a synthetic base graph
- `GENERATE_NODE_EMB` enables training for node embeddings 

### Parameters

- `CONV` is the type of convolution layer from Pytorch geometric to use (i.e. `gin`, `graphsaint_gcn`)
- `MINIBATCH` is the minibatching algorithm from Pytorch geometric to use (i.e. `NeighborSampler`, `GraphSaint`)
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


To generate `GIN` embeddings as in our paper, specify `CONV` = `gin` and `MINIBATCH` = `NeighborSampler`. To generate `GraphSaint` embeddings as in our work, specify `CONV` = `graphsaint_gcn` and `MINIBATCH` = `GraphSaint`.

The following examples are the parameters used to create the synthetic graphs described in the SubGNN paper.

#### Density

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

#### Cut Ratio

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

#### Component

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

#### Coreness

        DESIRED_PROPERTY = 'coreness'
        BASE_GRAPH_TYPE = 'duplication_divergence_graph' 
        SUBGRAPH_TYPE = 'plant'
        N_SUBGRAPHS = 30 # Number of subgraphs per coreness value; N_SUBGRAPHS = 30 results in ~250 total subgraphs
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
