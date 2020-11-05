# SubGNN Config files

Below we outline all of the hyperparameters that can be set in the config files. 

### `data`

`task`: which dataset to use (note that this should be the name of the folder in which the base graph edge list, subgraph labels, etc. live)

### `tb`
`tb_logging`: boolean designating whether or not to log to a tensorboard directory. We highly recommend setting this to true.

`dir`: the name of the tensorboard directory. 

`name`: the name of the run type. Results/models for a given run type will be stored at `config.PROJECT_ROOT/dir/name`.

### `optuna`

`opt_n_trials`: number of optuna trials to run (i.e. number of different hyperparameter combinations for the run type

`opt_n_cores`: number of cores to use for optuna. We always set this to 1. 

`monitor_metric`: the metric to use to assess which hyperparameter combination is best (e.g. `val_micro_f1`)

`opt_direction`: specifies the direction in which you want to optimize the metric (either `maximize` or `minimize`)

`sampler`: approach for sampling from hyperparameter ranges (e.g. `grid`, `random`, etc. See optuna for details.)

`pruning`: whether to stop unpromising trials at the early stages of the training. We always used pruning = false








