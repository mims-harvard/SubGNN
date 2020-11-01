# SubGNN
Repository for NeurIPS 2020 paper: [Subgraph Neural Networks](https://arxiv.org/abs/2006.10538)

## Install the Environment
We provide a yml file containing the necessary packages for SubGNN. Once you have [conda](https://docs.anaconda.com/anaconda/install/) installed, you can create an environment as follows:
```
conda env create --file SubGNN.yml 
```

## How to Train
To train SubGNN, you should first specify your project directory in `config.py`. This directory should include folders containing all datasets and will contain all tensorboard folders with model outputs. Then modify the config.json file for the appropriate dataset to set the tensorboard output directory and the hyperparameter search ranges, including which SubGNN channels (neighborhood, structure, or position) to turn on. Finally, train the model via the following: 

```
cd SubGNN
python train_config.py -config_path config_files/hpo_metab/metab_config.json
```

We use the `hpo_metab` dataset as as example, but you can easily run any of the datasets by passing in the appropriate config file. 

## How to Evaluate
Once you have trained SubGNN and selected the best hyperparameters on the validaation set, run the `test.py` script to re-train the model on 10 random seeds and evaluate on the test set:

```
cd SubGNN
python test.py \
-task hpo_metab \
-tb_dir NAME_OF_TENSORBOARD_FOLDER \
-tb_name NAME_OF_RUN_TYPE
-restoreModelPath PATH/TO/MODEL/LOCATION/WITH/BEST/HYPERPARAMETERS
```

Note that the restoreModelPath directory should contain a `.ckpt` file and a `hyperparams.json` file. The test performance on each random seed will be saved in `test_results.json` files in folders in the tensorboard directory specified by `tb_dir` and `tb_name`. The `experiment_results.json` file summarizes test performance across all random seeds.

## How to Cite
```
@article{alsentzer2020subgraph,
  title={Subgraph Neural Networks},
  author={Alsentzer, Emily and Finlayson, Samuel G and Li, Michelle M and Zitnik, Marinka},
  journal={Proceedings of Neural Information Processing Systems, NeurIPS},
  year={2020}
}
```
## Contact Us
Please open an issue or contact emilya@mit.edu with any questions.

