# General
import numpy as np
import random
import argparse
import os

import config_prepare_dataset as config
import preprocess
import model as mdl
import utils

# Pytorch
import torch
from torch_geometric.utils.convert import to_networkx, to_scipy_sparse_matrix
from torch_geometric.data import Data, DataLoader, NeighborSampler
if config.MINIBATCH == "GraphSaint": from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.utils import negative_sampling 

# Global Variables
log_f = open(str(config.DATASET_DIR / "node_emb.log"), "w")
all_data = None 
device = None
best_val_acc = -1
best_embeddings = None
best_model = None
all_losses = {}
eps = 10e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
best_hyperparameters = dict()
if device.type == 'cuda': print(torch.cuda.get_device_name(0))
if config.MINIBATCH == "NeighborSampler":
    all_hyperparameters = {'batch_size': config.POSSIBLE_BATCH_SIZES, 'hidden': config.POSSIBLE_HIDDEN, 'output': config.POSSIBLE_OUTPUT, 'lr': config.POSSIBLE_LR, 'wd': config.POSSIBLE_WD, 'nb_size': config.POSSIBLE_NB_SIZE, 'dropout': config.POSSIBLE_DROPOUT}
    curr_hyperparameters = {'batch_size': config.POSSIBLE_BATCH_SIZES[0], 'hidden': config.POSSIBLE_HIDDEN[0], 'output': config.POSSIBLE_OUTPUT[0], 'lr': config.POSSIBLE_LR[0], 'wd': config.POSSIBLE_WD[0], 'nb_size': config.POSSIBLE_NB_SIZE[0], 'dropout': config.POSSIBLE_DROPOUT[0]}
elif config.MINIBATCH == "GraphSaint":
    all_hyperparameters = {'batch_size': config.POSSIBLE_BATCH_SIZES, 'hidden': config.POSSIBLE_HIDDEN, 'output': config.POSSIBLE_OUTPUT, 'lr': config.POSSIBLE_LR, 'wd': config.POSSIBLE_WD, 'walk_length': config.POSSIBLE_WALK_LENGTH, 'num_steps': config.POSSIBLE_NUM_STEPS, 'dropout': config.POSSIBLE_DROPOUT}
    curr_hyperparameters = {'batch_size': config.POSSIBLE_BATCH_SIZES[0], 'hidden': config.POSSIBLE_HIDDEN[0], 'output': config.POSSIBLE_OUTPUT[0], 'lr': config.POSSIBLE_LR[0], 'wd': config.POSSIBLE_WD[0], 'walk_length': config.POSSIBLE_WALK_LENGTH[0], 'num_steps': config.POSSIBLE_NUM_STEPS[0], 'dropout': config.POSSIBLE_DROPOUT[0]}


def train(epoch, model, optimizer):

    global all_data, best_val_acc, best_embeddings, best_model, curr_hyperparameters, best_hyperparameters

    # Save predictions
    total_loss = 0
    roc_val = []
    ap_val = []
    f1_val = []
    acc_val = []

    # Minibatches
    if config.MINIBATCH == "NeighborSampler": 
        loader = NeighborSampler(all_data.edge_index, sizes = [curr_hyperparameters['nb_size']], batch_size = curr_hyperparameters['batch_size'], shuffle = True) 
    elif config.MINIBATCH == "GraphSaint": 
        all_data.num_classes = torch.tensor([2]) 
        loader = GraphSAINTRandomWalkSampler(all_data, batch_size=curr_hyperparameters['batch_size'], walk_length=curr_hyperparameters['walk_length'], num_steps=curr_hyperparameters['num_steps']) 

    # Iterate through minibatches
    for data in loader:
        
        if config.MINIBATCH == "NeighborSampler": data = preprocess.set_data(data, all_data, config.MINIBATCH) 
        curr_train_pos = data.edge_index[:, data.train_mask] 
        curr_train_neg = negative_sampling(curr_train_pos, num_neg_samples=curr_train_pos.size(1) // 4) 
        curr_train_total = torch.cat([curr_train_pos, curr_train_neg], dim=-1) 
        data.y = torch.zeros(curr_train_total.size(1)).float() 
        data.y[:curr_train_pos.size(1)] = 1. 

        # Perform training
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        curr_dot_embed = utils.el_dot(out, curr_train_total)
        loss = utils.calc_loss_both(data, curr_dot_embed)        
        if torch.isnan(loss) == False: 
            total_loss += loss
            loss.backward()
        optimizer.step()
        curr_train_pos_mask = torch.zeros(curr_train_total.size(1)).bool()
        curr_train_pos_mask[:curr_train_pos.size(1)] = 1
        curr_train_neg_mask = (curr_train_pos_mask == 0) 
        roc_score, ap_score, train_acc, train_f1 = utils.calc_roc_score(pred_all = curr_dot_embed.T[1], pos_edges = curr_train_pos_mask, neg_edges = curr_train_neg_mask) 
        print(">>>>>>Train: (ROC) ", roc_score, " (AP) ", ap_score, " (ACC) ", train_acc, " (F1) ", train_f1) 

        curr_val_pos = data.edge_index[:, data.val_mask]
        curr_val_neg = negative_sampling(curr_val_pos, num_neg_samples=curr_val_pos.size(1) // 4)
        curr_val_total = torch.cat([curr_val_pos, curr_val_neg], dim=-1) 
        curr_val_pos_mask = torch.zeros(curr_val_total.size(1)).bool()  
        curr_val_pos_mask[:curr_val_pos.size(1)] = 1
        curr_val_neg_mask = (curr_val_pos_mask == 0) 
        val_dot_embed = utils.el_dot(out, curr_val_total)
        data.y = torch.zeros(curr_val_total.size(1)).float()
        data.y[:curr_val_pos.size(1)] = 1. 
        roc_score, ap_score, val_acc, val_f1 = utils.calc_roc_score(pred_all = val_dot_embed.T[1], pos_edges = curr_val_pos_mask, neg_edges = curr_val_neg_mask) 
        roc_val.append(roc_score) 
        ap_val.append(ap_score) 
        acc_val.append(val_acc) 
        f1_val.append(val_f1) 
    res = "\t".join(["Epoch: %04d" % (epoch + 1), "train_loss = {:.5f}".format(total_loss), "val_roc = {:.5f}".format(np.mean(roc_val)), "val_ap = {:.5f}".format(np.mean(ap_val)), "val_f1 = {:.5f}".format(np.mean(f1_val)), "val_acc = {:.5f}".format(np.mean(acc_val))])
    print(res)
    log_f.write(res + "\n")

    # Save best model and parameters
    if best_val_acc <= np.mean(acc_val) + eps:
        best_val_acc = np.mean(acc_val)
        with open(str(config.DATASET_DIR / "best_model.pth"), 'wb') as f:
            torch.save(model.state_dict(), f)
        best_hyperparameters = curr_hyperparameters
        best_model = model

    return total_loss


def test(model):

    global all_data, best_embeddings, best_hyperparameters, all_losses

    model.load_state_dict(torch.load(str(config.DATASET_DIR / "best_model.pth")))
    model.to(device)
    model.eval()

    test_pos = all_data.edge_index[:, all_data.test_mask]
    test_neg = negative_sampling(test_pos, num_neg_samples=test_pos.size(1) // 4)
    test_total = torch.cat([test_pos, test_neg], dim=-1)
    test_pos_edges = torch.zeros(test_total.size(1)).bool()
    test_pos_edges[:test_pos.size(1)] = 1
    test_neg_edges = (test_pos_edges == 0)

    dot_embed = utils.el_dot(best_embeddings, test_total, test = True)
    roc_score, ap_score, test_acc, test_f1 = utils.calc_roc_score(pred_all = dot_embed, pos_edges = test_pos_edges.flatten(), neg_edges = test_neg_edges.flatten(), loss = all_losses, save_plots = config.DATASET_DIR / "train_plots.pdf")
    print('Test ROC score: {:.5f}'.format(roc_score))
    print('Test AP score: {:.5f}'.format(ap_score))
    print('Test Accuracy: {:.5f}'.format(test_acc))
    print('Test F1 score: {:.5f}'.format(test_f1))
    log_f.write('Test ROC score: {:.5f}\n'.format(roc_score))
    log_f.write('Test AP score: {:.5f}\n'.format(ap_score))
    log_f.write('Test Accuracy: {:.5f}\n'.format(test_acc))
    log_f.write('Test F1 score: {:.5f}\n'.format(test_f1))


def generate_emb():

    global all_data, best_embeddings, best_model, all_hyperparameters, curr_hyperparameters, best_hyperparameters, all_losses, device

    all_data = preprocess.read_graphs(str(config.DATASET_DIR / "edge_list.txt"))

    # Iterate through hyperparameter type (shuffled)
    shuffled_param_type = random.sample(all_hyperparameters.keys(), len(all_hyperparameters.keys()))
    for param_type in shuffled_param_type:

        # Iterate through hyperparameter values of the specified type (shuffled)
        shuffled_param_val = random.sample(all_hyperparameters[param_type], len(all_hyperparameters[param_type]))
        for param_val in shuffled_param_val:

            # Initiate current hyperparameter
            curr_hyperparameters[param_type] = param_val
            print(curr_hyperparameters)
            log_f.write(str(curr_hyperparameters) + "\n")

            # Set up
            model = mdl.TrainNet(all_data.x.shape[1], curr_hyperparameters['hidden'], curr_hyperparameters['output'], config.CONV.lower().split("_")[1], curr_hyperparameters['dropout']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = curr_hyperparameters['lr'], weight_decay = curr_hyperparameters['wd'])

            # Train model
            model.train()
            curr_losses = []
            for epoch in range(config.EPOCHS):
                loss = train(epoch, model, optimizer)
                curr_losses.append(loss)
            all_losses[";".join([str(v) for v in curr_hyperparameters.values()])] = curr_losses

            # Set up for next hyperparameter
            curr_hyperparameters[param_type] = best_hyperparameters[param_type]

    print("Best Hyperparameters: ", best_hyperparameters)
    print("Optimization finished!")
    log_f.write("Best Hyperparameters: %s \n" % best_hyperparameters)

    # Save best embeddings
    device = torch.device('cpu') 
    best_model = best_model.to(device)
    best_embeddings = utils.get_embeddings(best_model, all_data, device) 

    # Test
    test(best_model)

    # Save best embeddings
    torch.save(best_embeddings, config.DATASET_DIR / (config.CONV.lower() + "_embeddings.pth")) 
