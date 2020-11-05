# SubGNN Config files

Below we outline all of the hyperparameters that can be set in the config files. 

### data

`task`: which dataset to use (note that this should be the name of the folder in which the base graph edge list, subgraph labels, etc. live)

### tb
`tb_logging`: boolean designating whether or not to log to a tensorboard directory. We highly recommend setting this to true.

`dir`: the name of the tensorboard directory. 

`name`: the name of the run type. Results/models for a given run type will be stored at `config.PROJECT_ROOT/dir/name`.




