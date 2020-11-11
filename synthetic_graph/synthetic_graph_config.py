from pathlib import Path

# Random Seed
RANDOM_SEED = 42

# Output files
SAVE_GRAPH = "example_edge_list.txt"
SAVE_SUBGRAPHS = "example_subgraphs.pth"
SAVE_NODE_EMB = "example_node_emb.pth"
SAVE_NODE_EMB_LOG = "example_node_emb.log"
SAVE_NODE_EMB_PLOTS = "example_node_emb.pdf"
SAVE_MODEL = "example_save_model.pth"

# Flags
GENERATE_SYNTHETIC_G = True
GENERATE_NODE_EMB = True 

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

# Parameters for training node embeddings for base graph
POSSIBLE_BATCH_SIZES = [512, 1024]
POSSIBLE_HIDDEN = [128, 256]
POSSIBLE_OUTPUT = [64]
POSSIBLE_LR = [0.001, 0.005]
POSSIBLE_WD = [5e-4, 5e-5]
POSSIBLE_DROPOUT = [0.4, 0.5]
POSSIBLE_NB_SIZE = [1.0]
POSSIBLE_NUM_HOPS = [1]
EPOCHS = 100
