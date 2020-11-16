# General
import random
import numpy as np

# Pytorch
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid
from torch_geometric.data import Dataset

# Matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Sci-kit Learn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_loss_both(data, dot_pred): 
    """
    Calculate loss via link prediction

    Args
        - data (Data object): graph
        - dot_pred (tensor long, shape=(nodes, classes)): predictions calculated from dot product
    
    Return
        - loss (float): loss
    """

    loss = F.nll_loss(F.log_softmax(dot_pred.to(device), dim=-1), data.y.long())
    loss.requires_grad = True 
    return loss


def el_dot(embed, edges, test=False):
    """
    Calculate element-wise dot product for link prediction
   
    Args
        - embed (tensor): embedding
        - edges (tensor): list of edges

    Return
        - tensor of element-wise dot product
    """

    embed = embed.cpu().detach() 
    edges = edges.cpu().detach() 
    source = torch.index_select(embed, 0, edges[0, :]) 
    target = torch.index_select(embed, 0, edges[1, :]) 
    dots = torch.bmm(source.view(edges.shape[1], 1, embed.shape[1]), target.view(edges.shape[1], embed.shape[1], 1)) 
    dots = torch.sigmoid(np.squeeze(dots)) 
    if test: return dots 
    diff = np.squeeze(torch.ones((1, len(dots))) - dots)
    return torch.stack((diff, dots), 1)


def calc_roc_score(pred_all, pos_edges=[], neg_edges=[], true_all=[], save_plots="", loss = [], multi_class=False, labels=[], multilabel=False):
    """
    Calculate ROC score
    
    Args
        - pred_all 
        - pos_edges 
        - neg_edges 
        - true_all 
        - save_plots 
        - loss 
        - multi_class 
        - labels 
        - multilabel 
    
    Return
        - roc_auc 
        - ap_score 
        - acc 
        - f1 
    """ 
    if multi_class:
        if save_plots != "": 
            class_roc, class_ap, class_f1 = plot_roc_ap(true_all, pred_all, save_plots, loss = loss, labels = labels, multilabel = multilabel)

        roc_auc = roc_auc_score(true_all, pred_all, multi_class = 'ovr')

        if multilabel:
            pred_all = (pred_all > 0.5)
        else:
            true_all = torch.argmax(true_all, axis = 1)
            pred_all = torch.argmax(torch.tensor(pred_all), axis = 1)

        f1_micro = f1_score(true_all, pred_all, average = "micro")
        acc = accuracy_score(true_all, pred_all)
        
        if save_plots != "": return roc_auc, acc, f1_micro, class_roc, class_ap, class_f1
        return roc_auc, acc, f1_micro
    else:

        pred_pos = pred_all[pos_edges]
        pred_neg = pred_all[neg_edges]

        pred_all = torch.cat((pred_pos, pred_neg), 0).cpu().detach().numpy()
        true_all = torch.cat((torch.ones(len(pred_pos)), torch.zeros(len(pred_neg))), 0).cpu().detach().numpy()

        roc_auc = roc_auc_score(true_all, pred_all)
        ap_score = average_precision_score(true_all, pred_all)
        acc = accuracy_score(true_all, (pred_all > 0.5))
        f1 = f1_score(true_all, (pred_all > 0.5))

        if save_plots != "": plot_roc_ap(true_all, pred_all, save_plots, loss, multilabel = multilabel)
        return roc_auc, ap_score, acc, f1


######################################################
# Get best embeddings
#def get_embeddings(model, data_loader, device):
@torch.no_grad()
def get_embeddings(model, data, device):
    """
    Get best embeddings
    
    Args
        - model (torch object): best model
        - data (Data object): dataset
        - device (torch object): cpu or cuda
    
    Return
        - all_emb (tensor): best embedding for all nodes 
    """
    model.eval() 
    data = data.to(device) 
    all_emb = model(data.x, data.edge_index) 
    print(all_emb.shape)
    return all_emb 
