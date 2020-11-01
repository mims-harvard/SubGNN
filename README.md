# SubGNN
Repository for NeurIPS 2020 paper: [Subgraph Neural Networks](https://arxiv.org/abs/2006.10538)

## Install the Environment
We provide a yml file containing the necessary packages for SubGNN. Once you have [conda](https://docs.anaconda.com/anaconda/install/) installed, you can create an environment as follows:
```
conda env create --file SubGNN.yml 
```
## Datasets
We are releasing four new real-world datasets: HPO-NEURO, HPO-METAB, PPI-BP, and EM-USER. You can download these files from dropbox [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0). You should unzip the folder and set the project directory below to the path where you downloaded the data (e.g. `/PATH/TO/SubGNN_data`) We also provide a script to generate the DENSITY, CORENESS, COMPONENT, and CUTRATIO synthetic graphs featured in our paper. See the README in the synthetic graphs folder for more information on how to generate these synthetic datasets.

## How to Train
To train SubGNN, you should first specify your project directory via `PROJECT_ROOT` in `config.py`. This directory should include folders containing all datasets and will contain all tensorboard folders with model outputs. Then modify the config.json file for the appropriate dataset to set the tensorboard output directory and the hyperparameter search ranges, including which SubGNN channels (neighborhood, structure, or position) to turn on. Finally, train the model via the following: 

```
cd SubGNN
python train_config.py -config_path config_files/hpo_metab/metab_config.json
```

The model and asssociated hyperparameters will be saved in the tensorboard directory specified by `tb_dir` and `tb_name` in the config file. We use the `hpo_metab` dataset as as example, but you can easily run any of the datasets by passing in the appropriate config file. 

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

Note that the `restoreModelPath` directory should contain a `.ckpt` file and a `hyperparams.json` file. This command will create a tensorboard directory at `PROJECT_ROOT/tb_dir/tb_name` where `tb_dir` and `tb_name` are specified by the input parameters. The test performance on each random seed will be saved in `test_results.json` files in folders in this tensorboard directory. The `experiment_results.json` file summarizes test performance across all random seeds.

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
