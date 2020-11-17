# SubGNN
Repository for NeurIPS 2020 paper: [Subgraph Neural Networks](https://arxiv.org/abs/2006.10538)

Authors: [Emily Alsentzer*](https://emilyalsentzer.github.io/), [Sam Finlayson*](https://sgfin.github.io/), [Michelle Li](https://scholar.harvard.edu/michelleli), [Marinka Zitnik](https://zitniklab.hms.harvard.edu/)

[Project Website](https://zitniklab.hms.harvard.edu/projects/SubGNN/)

To use SubGNN, do the following:

- Install the environment
- Prepare data
- Modify `PROJECT_ROOT` in `config.py`
- Modify the appropriate `config.json` file
- Train and evaluate SubGNN

## Install the Environment
We provide a yml file containing the necessary packages for SubGNN. Once you have [conda](https://docs.anaconda.com/anaconda/install/) installed, you can create an environment as follows:
```
conda env create --file SubGNN.yml 
```
## Prepare data
Prepare data for SubGNN by either (1) downloading our provided datasets or following the steps in the `prepare_dataset` folder README to (2) generate synthetic datasets or (3) format your own data.

**Real-World Datasets:** We are releasing four new real-world datasets: HPO-NEURO, HPO-METAB, PPI-BP, and EM-USER. You can download these files from Dropbox [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0). You should unzip the folder and set the `PROJECT_ROOT` in `config.py` to the path where you downloaded the data (e.g. `/PATH/TO/SubGNN_data`). 

**Synthetic Datasets:** We also provide a script to generate the DENSITY, CORENESS, COMPONENT, and CUTRATIO synthetic graphs featured in our paper. See the [README](https://github.com/mims-harvard/SubGNN/tree/main/prepare_dataset#prepare-dataset) in the `prepare_dataset` folder for more information on how to generate these synthetic datasets.

**Your Own Data:** To use your own data with SubGNN, you will need an edge list file containing the edges of the base graph and a file containing the node ids of the subgraphs, their labels, and whether they are in the train/val/test splits. Then you will need to generate node embeddings and precompute similarity metrics. For more info on how to do this, refer to the [README](https://github.com/mims-harvard/SubGNN/tree/main/prepare_dataset#prepare-dataset) in the `prepare_dataset` folder.

## How to Train
To train SubGNN, you should first specify your project directory via `PROJECT_ROOT` in `config.py` if you haven't already. This directory should include folders containing all datasets and will ultimately contain all tensorboard folders with model outputs. Then, modify the `config.json` file for the appropriate dataset to set the tensorboard output directory and the hyperparameter search ranges, including which SubGNN channels (neighborhood, structure, or position) to turn on.  To learn more about the hyperparameters, go to the `README` in the `config_files` folder. Finally, train the model via the following: 

```
cd SubGNN
python train_config.py -config_path config_files/hpo_metab/metab_config.json
```

The model and asssociated hyperparameters will be saved in the tensorboard directory specified by `tb_dir` and `tb_name` in the config file. We use the `hpo_metab` dataset as as example, but you can easily run any of the datasets by passing in the appropriate config file. Note that, while you can also train the model via `train.py`, we highly recommend using `train_config.py` instead.

## How to Evaluate

### Re-train & test on 10 random seeds

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

### Test on single random seed

You can also evaluate the model on a single random seed. You can use `train.py` with the `-noTrain` and `-runTest` flags to restore a specific model and evaluate on test data. The results will be printed to the console.

```
cd SubGNN
python train.py \
-task hpo_metab \
-noTrain \
-runTest \
-no_save \ 
-restoreModelPath PATH/TO/SAVED/MODEL \ 
-restoreModelName CHECKPOINT_FILE_NAME.ckpt
```

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
