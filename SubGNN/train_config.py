# General
import numpy as np
import random
import argparse
import tqdm
import pickle
import json
import commentjson
import joblib
import os
import sys
import pathlib
from collections import OrderedDict
import random
import string

# Pytorch
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback

# Our Methods
import SubGNN as md
sys.path.insert(0, '..') # add config to path
import config


def parse_arguments():
    '''
    Read in the config file specifying all of the parameters
    '''
    parser = argparse.ArgumentParser(description="Learn subgraph embeddings")
    parser.add_argument('-config_path', type=str, default=None, help='Load config file')
    args = parser.parse_args()
    return args

def read_json(fname):
    '''
    Read in the json file specified by 'fname'
    '''
    with open(fname, 'rt') as handle:
        return commentjson.load(handle, object_hook=OrderedDict)

def get_optuna_suggest(param_dict, name, trial):
    '''
    Returns a suggested value for the hyperparameter specified by 'name' from the range of values in 'param_dict'

    name: string specifying hyperparameter
    trial: optuna trial
    param_dict: dictionary containing information about the hyperparameter (range of values & type of sampler)
            e.g.{
                    "type" : "suggest_categorical",
                    "args" : [[ 64, 128]]
                }
    '''
    module_name = param_dict['type'] # e.g. suggest_categorical, suggest_float
    args = [name]
    args.extend(param_dict['args']) # resulting list will look something like this ['batch_size', [ 64, 128]]
    if "kwargs" in param_dict:
        kwargs = dict(param_dict["kwargs"])
        return getattr(trial, module_name)(*args, **kwargs) 
    else:
        return getattr(trial, module_name)(*args)

def get_hyperparams_optuna(run_config, trial):
    '''
    Converts the fixed and variable hyperparameters in the run config to a dictionary of the final hyperparameters

    Returns: hyp_fix - dictionary where key is the hyperparameter name (e.g. batch_size) and value is the hyperparameter value
    '''
    #initialize the dict with the fixed hyperparameters
    hyp_fix = dict(run_config["hyperparams_fix"])

    # update the dict with variable value hyperparameters by sampling a hyperparameter value from the range specified in the run_config
    hyp_optuna = {k:get_optuna_suggest(run_config["hyperparams_optuna"][k], k, trial) for k in dict(run_config["hyperparams_optuna"]).keys()}
    hyp_fix.update(hyp_optuna)
    return hyp_fix

def build_model(run_config, trial = None):
    '''
    Creates SubGNN from the hyperparameters specified in the run config
    '''
    # get hyperparameters for the current trial
    hyperparameters = get_hyperparams_optuna(run_config, trial)

    # Set seeds for reproducibility
    torch.manual_seed(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    torch.cuda.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed_all(hyperparameters['seed']) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize SubGNN
    model = md.SubGNN(hyperparameters, run_config["graph_path"], \
        run_config["subgraphs_path"], run_config["embedding_path"], \
        run_config["similarities_path"], run_config["shortest_paths_path"], run_config['degree_sequence_path'], run_config['ego_graph_path'])
    return model, hyperparameters

def build_trainer(run_config, hyperparameters, trial = None):
    '''
    Set up optuna trainer
    '''

    if 'progress_bar_refresh_rate' in hyperparameters:
        p_refresh = hyperparameters['progress_bar_refresh_rate']
    else:
        p_refresh = 5

    # set epochs, gpus, gradient clipping, etc. 
    # if 'no_gpu' in run config, then use CPU
    trainer_kwargs={'max_epochs': hyperparameters['max_epochs'],
                    "gpus": 0 if 'no_gpu' in run_config else 1,
                    "num_sanity_val_steps":0,
                    "progress_bar_refresh_rate":p_refresh,
                    "gradient_clip_val": hyperparameters['grad_clip']
                    }

    # set auto learning rate finder param
    if 'auto_lr_find' in hyperparameters and hyperparameters['auto_lr_find']:
        trainer_kwargs['auto_lr_find'] = hyperparameters['auto_lr_find']

    # Create tensorboard logger
    lgdir = os.path.join(run_config['tb']['dir_full'], run_config['tb']['name'])
    if not os.path.exists(lgdir):
        os.makedirs(lgdir)
    logger = TensorBoardLogger(run_config['tb']['dir_full'], name=run_config['tb']['name'], version="version_"+ str(random.randint(0, 10000000)))
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir)
    print("Tensorboard logging at ", logger.log_dir)
    trainer_kwargs["logger"] = logger


    # Save top three model checkpoints
    trainer_kwargs["checkpoint_callback"] = ModelCheckpoint(
            filepath= os.path.join(logger.log_dir, "{epoch}-{val_micro_f1:.2f}-{val_acc:.2f}-{val_auroc:.2f}"),
            save_top_k = 3,
            verbose=True,
            monitor=run_config['optuna']['monitor_metric'],
            mode='max'
            )

    # if we use pruning, use the pytorch lightning pruning callback
    if run_config["optuna"]['pruning']:
        trainer_kwargs['early_stop_callback'] = PyTorchLightningPruningCallback(trial, monitor=run_config['optuna']['monitor_metric'])

    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer, trainer_kwargs, logger.log_dir  

def train_model(run_config, trial = None):
    '''
    Train a single model whose hyperparameters are specified in the run config
    
    Returns the max (or min) metric specified by 'monitor_metric' in the run config
    '''

    # get model and hyperparameter dict
    model, hyperparameters = build_model(run_config, trial)

    # build optuna trainer
    trainer, trainer_kwargs, results_path = build_trainer(run_config, hyperparameters, trial)

    # dump hyperparameters to results dir
    hparam_file = open(os.path.join(results_path, "hyperparams.json"),"w")
    hparam_file.write(json.dumps(hyperparameters, indent=4))
    hparam_file.close()
    
    # dump trainer args to results dir
    tkwarg_file = open(os.path.join(results_path, "trainer_kwargs.json"),"w")
    pop_keys = [key for key in ['logger','profiler','early_stop_callback','checkpoint_callback'] if key in trainer_kwargs.keys()]
    [trainer_kwargs.pop(key) for key in pop_keys]
    tkwarg_file.write(json.dumps(trainer_kwargs, indent=4))
    tkwarg_file.close()

    # train the model
    trainer.fit(model)        
        
    # write results to the results dir
    if results_path is not None:
        hparam_file = open(os.path.join(results_path, "final_metric_scores.json"),"w")
        results_serializable = {k:float(v) for k,v in model.metric_scores[-1].items()}
        hparam_file.write(json.dumps(results_serializable, indent=4))
        hparam_file.close()
    
    # return the max (or min) metric specified by 'monitor_metric' in the run config
    all_scores = [score[run_config['optuna']['monitor_metric']].numpy() for score in model.metric_scores]
    if run_config['optuna']['opt_direction'] == "maximize":
        return(np.max(all_scores))
    else:
        return(np.min(all_scores))

def main():
    '''
    Perform an optuna run according to the hyperparameters and directory locations specified in 'config_path'
    '''
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()

    # read in config file
    run_config = read_json(args.config_path)

    ## Set paths to data
    task = run_config['data']['task']
    embedding_type = run_config['hyperparams_fix']['embedding_type']
    
    # paths to subgraphs, edge list, and shortest paths between all nodes in the graph
    run_config["subgraphs_path"] = os.path.join(task, "subgraphs.pth")
    run_config["graph_path"] = os.path.join(task, "edge_list.txt")
    run_config['shortest_paths_path'] = os.path.join(task, "shortest_path_matrix.npy")
    run_config['degree_sequence_path'] = os.path.join(task, "degree_sequence.txt")
    run_config['ego_graph_path'] = os.path.join(task, "ego_graphs.txt")

    #directory where similarity calculations will be stored
    run_config["similarities_path"] = os.path.join(task, "similarities/")

    # get location of node embeddings
    if embedding_type == 'gin':
        run_config["embedding_path"] = os.path.join(task, "gin_embeddings.pth")
    elif embedding_type == 'graphsaint':
        run_config["embedding_path"] = os.path.join(task, "graphsaint_gcn_embeddings.pth")
    else:
        raise NotImplementedError
    
    # create a tensorboard directory in the folder specified by dir in the PROJECT ROOT folder
    if 'local' in run_config['tb'] and run_config['tb']['local']:
        run_config['tb']['dir_full'] = run_config['tb']['dir']
    else:
        run_config['tb']['dir_full'] = os.path.join(config.PROJECT_ROOT, run_config['tb']['dir'])
    ntrials = run_config['optuna']['opt_n_trials']
    print(f'Running {ntrials} Trials of optuna')

    if run_config['optuna']['pruning']:
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = None

    # the complete study path is the tensorboard directory + the study name
    run_config['study_path'] = os.path.join(run_config['tb']['dir_full'], run_config['tb']['name'])
    print("Logging to ", run_config['study_path'])
    pathlib.Path(run_config['study_path']).mkdir(parents=True, exist_ok=True)

    # get database file
    db_file = os.path.join(run_config['study_path'], 'optuna_study_sqlite.db')

    # specify sampler
    if run_config['optuna']['sampler'] == "grid" and "grid_search_space" in run_config['optuna']:
        sampler = optuna.samplers.GridSampler(run_config['optuna']['grid_search_space'])
    elif run_config['optuna']['sampler'] == "tpe":
        sampler = optuna.samplers.TPESampler()
    elif run_config['optuna']['sampler'] == "random":
        sampler = optuna.samplers.RandomSampler()
    
    # create an optuna study with the specified sampler, pruner, direction (e.g. maximize)
    # A SQLlite database is used to keep track of results
    # Will load in existing study if one exists
    study = optuna.create_study(direction=run_config['optuna']['opt_direction'],
                                sampler=sampler,
                                pruner=pruner,
                                storage= 'sqlite:///' + db_file,
                                study_name=run_config['study_path'],
                                load_if_exists=True)
    
    study.optimize(lambda trial: train_model(run_config, trial), n_trials=run_config['optuna']['opt_n_trials'], n_jobs =run_config['optuna']['opt_n_cores'])
    
    optuna_results_path = os.path.join(run_config['study_path'], 'optuna_study.pkl')
    print("Saving Study Results to", optuna_results_path)
    joblib.dump(study, optuna_results_path)

    print(study.best_params)
    

if __name__ == "__main__":
    main()