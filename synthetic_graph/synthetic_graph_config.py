from pathlib import Path

PROJECT_ROOT = Path('.')

RANDOM_SEED = 42

# Output files
SAVE_GRAPH = "example_edge_list.txt"
SAVE_SUBGRAPHS = "example_subgraphs.pth"

# Parameters for generating subgraphs with specific properties
DESIRED_PROPERTY = "density"
BASE_GRAPH_TYPE = "barabasi_albert"
SUBGRAPH_TYPE = "bfs"
N_SUBGRAPHS = 250
N_CONNECTED_COMPONENTS = 1 
N_SUBGRAPH_NODES = 20
FEATURES_TYPE = "one_hot"
N = 1000
P = 0.5
Q = 0
M = 5
N_BINS = 3
SUBGRAPH_GENERATOR = "complete"
MODIFY_GRAPH_FOR_PROPERTIES = True

DENSITY_EPSILON = 0.01
DENSITY_RANGE = [0.05, 0.25, 0.45]
CUT_RATIO_EPSILON = 0.001 
CUT_RATIO_RANGE = [0.005, 0.0125, 0.02] 
K_HOPS_RANGE = [0.12, 0.5, 1.0]
BA_P_RANGE = [0.1, 0.5, 0.9]
CC_RANGE = [1, 1, 1, 1, 5, 6, 7, 8, 9, 10]

MAX_TRIES = 100

