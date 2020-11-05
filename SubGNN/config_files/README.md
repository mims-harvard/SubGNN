# SubGNN Config Files

Below we outline all of the hyperparameters that can be set in the config files. 

### `data`

Specify the dataset.

`task`: which dataset to use (note that this should be the name of the folder in which the base graph edge list, subgraph labels, etc. live)

### `tb`

Specify tensorboard logging.

`tb_logging`: boolean designating whether or not to log to a tensorboard directory. We highly recommend setting this to true.

`dir`: the name of the tensorboard directory. 

`name`: the name of the run type. Results/models for a given run type will be stored at `config.PROJECT_ROOT/dir/name`.

### `optuna`

Specify settings for the optuna study

`opt_n_trials`: number of optuna trials to run (i.e. number of different hyperparameter combinations for the run type

`opt_n_cores`: number of cores to use for optuna. We always set this to 1. 

`monitor_metric`: the metric to use to assess which hyperparameter combination is best (e.g. `val_micro_f1`)

`opt_direction`: specifies the direction in which you want to optimize the metric (either `maximize` or `minimize`)

`sampler`: approach for sampling from hyperparameter ranges (e.g. `grid`, `random`, etc. See optuna for details.)

`pruning`: whether to stop unpromising trials at the early stages of the training. We always used pruning = false


### `hyperparams_fix` and `hyperparams_optuna`

`hyperparams_fix` specifies the fixed hyperparameters, and `hyperparams_optuna` specifies which hyperparameters to search over. 

`compute_similarities`: boolean specifying whether to recompute similarity calculations, even if they've already been precomputed and saved to a file

`n_processes`: number of processes to use for multi-processing code (used to precompute similarities efficiently)

`use_mpn_projection`: boolean specifying whether to push the aggregated messages through a projection layer in the message passing layer. This is always set to true in our runs. 

`seed`: random seed

`max_epochs`: maximum number of epochs to train the model

`use_neighborhood`: whether to use the neighborhood channel

`use_structure`: whether to use the structure channel

`use_position`: whether to use the position channel

`ff_attn`: boolean specifying whether to use feed forward attention in the read out section of SubGNN. This is always false in our runs.

`node_embed_size`: node embedding dimension

`embedding_type`: type of node embeddings (either "gin" or "graphsaint_gcn")

`freeze_node_embeds`: boolean whether or not to freeze (or train) the node embeddings

`resample_anchor_patches`: boolean specifying whether to resample anchor patches each epoch. This is always false in our runs.

`sample_walk_len`: the length of the random walk used to sample structure channel anchor patches

`structure_patch_type`: approach used to sample structure anchor patches (either `triangular_random_walk` or `ego_graph`, but we always use `triangular_random_walk` in our runs)

`n_triangular_walks`: number or triangular random walks used to embed the structure anchor patches

`random_walk_len`: length of the random walk used to embed the structure channel anchor patches

`lstm_aggregator`: approach for aggregating LSTM hidden states (either `last` or `sum`). We default to `last` in our runs.

`rw_beta`: triangular random walk beta parameter determines whether triangles or non-triangles will be privi-leged during sampling

`max_sim_epochs`: integer that controls how many structure patches are initially sampled during precomputing. We default to 5. This parameter is only important if `resample_anchor_patches` is true. 









