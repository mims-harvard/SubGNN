# General
import numpy as np
import random
import argparse
import tqdm
import pickle
import json
import joblib
import os
import time
import sys
import pathlib
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

'''
There are several options for running `train.py`:

(1) Specify a model path via restoreModelPath. This script will use the hyperparameters at that path to train a model. 
(2) Specify opt_n_trials != None and restoreModelPath == None. This script will use the hyperparameter ranges set 
in the `get_hyperparams_optuma` function to run optuna trials.
(3) Specify opt_n_trials == None and restoreModelPath == None. This script will use the hyperparameters in the
`get_hyperparams` function to train/test the model.
'''

###################################################
# Parse arguments
def parse_arguments():
    '''
    Collect and parse arguments to script
    '''
    parser = argparse.ArgumentParser(description="Learn subgraph embeddings")
    parser.add_argument('-embedding_path', type=str,  help='Directory where node embeddings are saved')
    parser.add_argument('-subgraphs_path', type=str,  help='File where subgraphs are saved')
    parser.add_argument('-shortest_paths_path', type=str,  help='File where subgraphs are saved')
    parser.add_argument('-graph_path', type=str, help='File where graph is saved')
    parser.add_argument('-similarities_path', type=str, help='File where graph is saved')
    parser.add_argument('-task', type=str, help='Task name (e.g. hpo_metab)')

    # Max Epochs
    parser.add_argument("-max_epochs", type=int, default=None, help="Max number of epochs to train")
    parser.add_argument("-seed", type=int, default=None, help="Random Seed")

    # Log
    parser.add_argument('-log_path', type=str, default=None, help='Place to store results. By default, use tensorboard directory unless -no_save.')
    parser.add_argument('-no_save', action='store_true', help='Makes model not save any specifications.')
    parser.add_argument('-print_train_times', action='store_true', help='Print train times.')

    # Tensorboard Arguments
    parser.add_argument('-tb_logging', action='store_true', help='Log to Tensorboard')
    parser.add_argument('-tb_dir', type=str, default="tensorboard", help='Directory for Tensorboard Logs')
    parser.add_argument('-tb_name', type=str, default="sg", help='Base Model Name for Tensorboard Log')
    parser.add_argument('-tb_version', help='Version Name for Tensorboard Log. (By default, created automatically.)')

    # Checkpoint
    parser.add_argument('-no_checkpointing', action='store_true', help='Specifies not to do model checkpointing.')
    parser.add_argument('-checkpoint_k', type=int, default=3, help='Frequency with which to save model checkpoints')
    parser.add_argument('-monitor_metric', type=str, default='val_micro_f1', help='Metric to monitor for checkpointing/stopping')

    # Optuma
    parser.add_argument("-opt_n_trials", type=int, default=None, help="Number of optuma trials to run")
    parser.add_argument("-opt_n_cores", type=int, default=-1, help="Number of cores (-1 = all available)")
    parser.add_argument("-opt_prune", action='store_true', help="Prune trials early if not promising")
    parser.add_argument("-grid_search", action='store_true', help="Grid search")

    #Debug
    parser.add_argument('-debug_mode', action='store_true', help='Plot gradients + GPU usage')
    parser.add_argument('-subset_data', action='store_true', help='Subset data to one batch per dataset')

    # Restore Model
    parser.add_argument('-restoreModelPath', type=str, default=None, help='Parent directory of model, hparams, kwargs')
    parser.add_argument('-restoreModelName', type=str, default=None, help='Name of model to restore')

    # Test set
    parser.add_argument('-runTest', action='store_true', help='Run on the test set')
    parser.add_argument('-noTrain', action='store_true', help='No training')
    
    args = parser.parse_args()
    return args

###################################################
# Set Hyperparameters
# TODO: change the values here if you run this script

def get_hyperparams(args):
    '''
    You, the user, should change these hyperparameters to best suit your model/run
    NOTE: These hyperparameters are only used if args.opt_n_trials is None and restoreModelPath is None
    '''
    hyperparameters = {
        "max_epochs": 200,
        "use_neighborhood": True,
        "use_structure": True,
        "use_position": True,
        "seed": 3,
        "node_embed_size": 128,
        "structure_patch_type": "triangular_random_walk",
        "lstm_aggregator": "last",
        "n_processes": 4,
        "resample_anchor_patches": False,
        "freeze_node_embeds": False,
        "use_mpn_projection": True,
        "print_train_times": False,
        "compute_similarities": False, 
        "sample_walk_len": 50,
        "n_triangular_walks": 5,
        "random_walk_len": 10,
        "rw_beta": 0.65,
        "set2set": False,
        "ff_attn": False,
        "batch_size": 64,
        "learning_rate": 0.00025420762516423353,
        "grad_clip": 0.2160947806012501,
        "n_layers": 1,
        "neigh_sample_border_size": 1,
        "n_anchor_patches_pos_out": 123,
        "n_anchor_patches_pos_in": 34,
        "n_anchor_patches_N_in": 19,
        "n_anchor_patches_N_out": 69,
        "n_anchor_patches_structure": 37,
        "linear_hidden_dim_1": 64,
        "linear_hidden_dim_2": 32,
        "lstm_dropout": 0.21923625197416907,
        "lstm_n_layers": 2,
        "lin_dropout": 0.04617609616314509,
        "cc_aggregator": "max",
        "trainable_cc": True,
        "auto_lr_find": True
    }

    return hyperparameters

def get_hyperparams_optuma(args, trial):
    '''
    If you specify args.opt_n_trials != None (and restoreModelPath == None), then the script will use the hyperparameter ranges
    specified here to train/test the model
    '''
    hyperparameters={'seed': 42,
            'batch_size': trial.suggest_int('batch_size', 64,150),
            'learning_rate':  trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),  #learning rate
            'grad_clip': trial.suggest_float('grad_clip', 0, 0.5), #gradient clipping
            'max_epochs': args.max_epochs, #max number of epochs
            'node_embed_size': 32, # dim of node embedding 
            'n_layers': trial.suggest_int('gamma_shortest_max_distance_N', 1,5), # number of layers
            'n_anchor_patches_pos_in': trial.suggest_int('n_anchor_patches_pos_in', 25, 75), # number of anchor patches (P, INTERNAL)
            'n_anchor_patches_pos_out': trial.suggest_int('n_anchor_patches_pos_out', 50, 200),  # number of anchor patches (P, BORDER)
            'n_anchor_patches_N_in': trial.suggest_int('n_anchor_patches_N_in', 10, 25), # number of anchor patches (N, INTERNAL)
            'n_anchor_patches_N_out': trial.suggest_int('n_anchor_patches_N_out', 25, 75),  # number of anchor patches (N, BORDER)
            'n_anchor_patches_structure': trial.suggest_int('n_anchor_patches_structure', 15, 40),  # number of anchor patches (S, INTERNAL & BORDER)
            'neigh_sample_border_size': trial.suggest_int('neigh_sample_border_size', 1,2), 
            'linear_hidden_dim_1': trial.suggest_int('linear_hidden_dim', 16, 96), 
            'linear_hidden_dim_2': trial.suggest_int('linear_hidden_dim', 16, 96), 
            'n_triangular_walks': trial.suggest_int('n_triangular_walks', 5, 15), 
            'random_walk_len': trial.suggest_int('random_walk_len', 18, 26), 
            'sample_walk_len': trial.suggest_int('sample_walk_len', 18, 26), 
            'rw_beta': trial.suggest_float('rw_beta', 0.1, 0.9), #triangular random walk parameter, beta
            'lstm_aggregator': 'last',
            'lstm_dropout': trial.suggest_float('lstm_dropout', 0.0, 0.4),
            'lstm_n_layers': trial.suggest_int('lstm_n_layers', 1, 2), #number of layers in LSTM used for embedding structural anchor patches
            'n_processes': 4, # multiprocessing 
            'lin_dropout': trial.suggest_float('lin_dropout', 0.0, 0.6),
            'resample_anchor_patches': False,
            'compute_similarities': False,
            'use_mpn_projection':True,
            'use_neighborhood': True,
            'use_structure': False,
            'use_position': False,
            'cc_aggregator': trial.suggest_categorical('cc_aggregator', ['sum', 'max']), #approach for aggregating node embeddings in components
            'trainable_cc': trial.suggest_categorical('trainable_cc', [True, False]),
            'freeze_node_embeds':False,
            'print_train_times':args.print_train_times
            }
    return hyperparameters


###################################################

def get_paths(args, hyperparameters):
    '''
    Returns the paths to data (subgraphs, embeddings, similarity calculations, etc)
    '''
    if args.task is not None:
        task = args.task
        embedding_type = hyperparameters['embedding_type']
        
        # paths to subgraphs, edge list, and shortest paths between all nodes in the graph
        subgraphs_path = os.path.join(task, "subgraphs.pth")
        graph_path = os.path.join(task, "edge_list.txt")
        shortest_paths_path = os.path.join(task, "shortest_path_matrix.npy")
        degree_sequence_path = os.path.join(task, "degree_sequence.txt")
        ego_graph_path = os.path.join(task, "ego_graphs.txt")

        #directory where similarity calculations will be stored
        similarities_path = os.path.join(task, "similarities/")

        # get location of node embeddings
        if embedding_type == 'gin':
            embedding_path = os.path.join(task, "gin_embeddings.pth")
        elif embedding_type == 'graphsaint':
            embedding_path = os.path.join(task, "graphsaint_gcn_embeddings.pth")
        else:
            raise NotImplementedError

        return graph_path, subgraphs_path, embedding_path, similarities_path, shortest_paths_path, degree_sequence_path, ego_graph_path
    else:
        return args.graph_path, args.subgraphs_path, args.embedding_path, args.similarities_path, args.shortest_paths_path, args.degree_sequence_path, args.ego_graph_path
    
def build_model(args, trial = None):
    '''
    Creates SubGNN from the hyperparameters specifid in either (1) restoreModelPath, (2) get_hyperparams_optuma, or (3) get_hyperparams
    '''
  
    #get hyperparameters
    if args.restoreModelPath is not None: # load in hyperparameters from file
        print("Loading Hyperparams")
        with open(os.path.join(args.restoreModelPath, "hyperparams.json")) as data_file: 
            hyperparameters = json.load(data_file)
        if args.max_epochs:
            hyperparameters['max_epochs'] = args.max_epochs
    elif trial is not None: #select hyperparams from ranges specified in trial
        hyperparameters = get_hyperparams_optuma(args, trial)
    else: #get hyperparams from passed in args
        hyperparameters = get_hyperparams(args)

    # set subset_data
    if args.subset_data:
        hyperparameters['subset_data'] = True

    # set seed
    if hasattr(args,"seed") and args.seed is not None:
        hyperparameters['seed'] = args.seed

    # set for reproducibility
    torch.manual_seed(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    torch.cuda.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed_all(hyperparameters['seed']) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get locations of file paths & instantiate model
    graph_path, subgraphs_path, embedding_path, similarities_path, shortest_paths_path, degree_dict_path, ego_graph_path = get_paths(args, hyperparameters)
    model = md.SubGNN(hyperparameters, graph_path, subgraphs_path, embedding_path, similarities_path, shortest_paths_path, degree_dict_path, ego_graph_path)

    # Restore Previous Weights, if relevant
    if args.restoreModelName:
        checkpoint_path = os.path.join(args.restoreModelPath, args.restoreModelName)
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, torch.device('cpu') )
        else:
            checkpoint = torch.load(checkpoint_path)
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        model.load_state_dict(pretrain_dict)

    return model, hyperparameters

def build_trainer(args, hyperparameters, trial = None):
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
                    "gpus":1,
                    "num_sanity_val_steps":0,
                    "progress_bar_refresh_rate":p_refresh,
                    "gradient_clip_val": hyperparameters['grad_clip']
                    }

    # set auto learning rate finder param
    if 'auto_lr_find' in hyperparameters and hyperparameters['auto_lr_find']:
        trainer_kwargs['auto_lr_find'] = hyperparameters['auto_lr_find']
    
        
    # Create tensorboard logger
    if not args.no_save and args.tb_logging:
        lgdir = os.path.join(args.tb_dir, args.tb_name)
        if not os.path.exists(lgdir):
            os.makedirs(lgdir)
        if args.tb_version is not None:
            tb_version = args.tb_version
        else:
            tb_version = "version_"+ str(random.randint(0, 10000000))

        logger = TensorBoardLogger(args.tb_dir, name=args.tb_name, version=tb_version)
        if not os.path.exists(logger.log_dir):
            os.makedirs(logger.log_dir)
        print("Tensorboard logging at ", logger.log_dir)
        trainer_kwargs["logger"] = logger
        
    # set up model saving
    results_path = None
    if not args.no_save:
        if args.log_path:
            results_path = args.log_path
        elif args.tb_logging:
            results_path = logger.log_dir
        else:
            raise Exception('No results path has been specified.')
     
        if (not args.no_save) and (not args.no_checkpointing):
            trainer_kwargs["checkpoint_callback"] = ModelCheckpoint(
                    filepath= os.path.join(results_path, "{epoch}-{val_micro_f1:.2f}-{val_acc:.2f}-{val_auroc:.2f}"),
                    save_top_k = args.checkpoint_k,
                    verbose=True,
                    monitor=args.monitor_metric,
                    mode='max'
                    )

        if trial is not None and args.opt_prune:
            trainer_kwargs['early_stop_callback'] = PyTorchLightningPruningCallback(trial, monitor=args.monitor_metric)

    # enable debug mode 
    if args.debug_mode:
        print("\n**** DEBUG MODE ON! ****\n")
        trainer_kwargs["track_grad_norm"] = 2
        trainer_kwargs["log_gpu_memory"] = True
        trainer_kwargs['print_nan_grads'] = False

        if not args.no_save:
            profile_path = os.path.join(results_path, "profiler.log")
            print("Profiling to ", profile_path)
            trainer_kwargs["profiler"] = AdvancedProfiler(output_filename=profile_path)
        else:
            trainer_kwargs["profiler"] = AdvancedProfiler()

    # set GPU availability
    if not torch.cuda.is_available():
        trainer_kwargs['gpus'] = 0

    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer, trainer_kwargs, results_path  

def train_model(args, trial = None):
    '''
    Train a single model whose hyperparameters are specified in the run config
    
    Returns the max (or min) metric specified by 'monitor_metric' in the run config
    '''
    model, hyperparameters = build_model(args, trial)
    trainer, trainer_kwargs, results_path = build_trainer(args, hyperparameters, trial)
    random.seed(hyperparameters['seed'])


    # save hyperparams and trainer kwargs to file
    if results_path is not None:
        hparam_file = open(os.path.join(results_path, "hyperparams.json"),"w")
        hparam_file.write(json.dumps(hyperparameters, indent=4))
        hparam_file.close()
        
        tkwarg_file = open(os.path.join(results_path, "trainer_kwargs.json"),"w")
        pop_keys = [key for key in ['logger','profiler','early_stop_callback','checkpoint_callback'] if key in trainer_kwargs.keys()]
        [trainer_kwargs.pop(key) for key in pop_keys]
        tkwarg_file.write(json.dumps(trainer_kwargs, indent=4))
        tkwarg_file.close()

    # optionally train the model
    if not args.noTrain:
        trainer.fit(model)

    # optionally test the model
    if args.runTest or args.noTrain:
        # reproducibility
        torch.manual_seed(hyperparameters['seed'])
        np.random.seed(hyperparameters['seed'])
        torch.cuda.manual_seed(hyperparameters['seed'])
        torch.cuda.manual_seed_all(hyperparameters['seed']) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if not args.no_checkpointing:
            for file in os.listdir(results_path):
                if file.endswith(".ckpt") and file.startswith("epoch"):
                    print(f"Loading model {file}")
                    if not torch.cuda.is_available():
                        checkpoint = torch.load(os.path.join(results_path, file), torch.device('cpu') )
                    else:
                        checkpoint = torch.load(os.path.join(results_path, file))
                    model_dict = model.state_dict()
                    pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
                    model.load_state_dict(pretrain_dict)
        trainer.test(model)

    # save results
    if results_path is not None:
        scores_file = open(os.path.join(results_path, "final_metric_scores.json"),"w")
        results_serializable = {k:float(v) for k,v in model.metric_scores[-1].items()}
        scores_file.write(json.dumps(results_serializable, indent=4))
        scores_file.close()

        if args.runTest:
            scores_file = open(os.path.join(results_path, "test_results.json"),"w")
            results_serializable = {k:float(v) for k,v in model.test_results.items()}
            scores_file.write(json.dumps(results_serializable, indent=4))
            scores_file.close()
    
    # print results
    if args.runTest:
        print(model.test_results)
        return model.test_results
    elif args.noTrain:
        print(model.test_results)
        return model.test_results 
    else:
        all_scores = [score[args.monitor_metric].numpy() for score in model.metric_scores]
        if args.monitor_metric == "val_loss":
            return(np.min(all_scores))
        else:
            return(np.max(all_scores))

def main(args):
    torch.autograd.set_detect_anomaly(True)

    # specify tensorboard directory
    if args.tb_dir is not None:
        args.tb_dir = os.path.join(config.PROJECT_ROOT, args.tb_dir)

    # if args.opt_n_trials is None, then we use either read in hparams from file or use the hyperparameters in get_hyperparams
    if args.opt_n_trials is None:
        return train_model(args)
    else:
        print(f'Running {args.opt_n_trials} Trials of optuna')
        if args.opt_prune:
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = None

        if args.monitor_metric == 'val_loss':
            direction = "minimize"
        else:
            direction = "maximize"

        if args.log_path:
            study_path = args.log_path
        elif args.tb_logging:
            study_path = os.path.join(args.tb_dir, args.tb_name)

        print("Logging to ", study_path)
        db_file = os.path.join(study_path, 'optuma_study_sqlite.db')
        pathlib.Path(study_path).mkdir(parents=True, exist_ok=True)

        # set up optuna study
        if args.grid_search:
            search_space = {
                'neigh_sample_border_size': [1,2],
                'gamma_shortest_max_distance_P': [3,4,5,6]
            }
            sampler = optuna.samplers.GridSampler(search_space)
        else:
            sampler = optuna.samplers.RandomSampler()

        study = optuna.create_study(direction=direction,
                                    sampler=sampler,
                                    pruner=pruner,
                                    storage= 'sqlite:///' + db_file,
                                    study_name=study_path,
                                    load_if_exists=True)
        
        study.optimize(lambda trial: train_model(args, trial), n_trials=args.opt_n_trials, n_jobs = args.opt_n_cores)
        
        
        optuma_results_path = os.path.join(study_path, 'optuna_study.pkl')
        print("Saving Study Results to", optuma_results_path)
        joblib.dump(study, optuma_results_path)

        print(study.best_params)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
