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


def plot_roc_ap(y_true, y_pred, save_plots, loss = {}, labels = [], multilabel = False):
    with PdfPages(save_plots) as pdf:

        # ROC
        fpr = dict()
        tpr = dict()
        roc = dict()
        if len(labels) > 0: # Multiclass classification
            for c in range(y_true.shape[1]):
                fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_pred[:, c])
                roc[c] = roc_auc_score(y_true[:, c], y_pred[:, c])
                plt.plot(fpr[c], tpr[c], label = str(labels[c]) + " (area = {:.5f})".format(roc[c]))
                print("[ROC] " + str(labels[c]) + ": {:.5f}".format(roc[c]))
        else: # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc = roc_auc_score(y_true, y_pred)
            plt.plot(fpr, tpr, label = "ROC = {:.5f}".format(roc))
            plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.title("ROC")
        pdf.savefig()
        plt.close()

        # Precision-Recall curve
        precision = dict()
        recall = dict()
        ap = dict()
        if len(labels) > 0: # Multiclass classification
            for c in range(y_true.shape[1]):
                precision[c], recall[c], _ = precision_recall_curve(y_true[:, c], y_pred[:, c])
                ap[c] = average_precision_score(y_true[:, c], y_pred[:, c])
                plt.plot(recall[c], precision[c], label = str(labels[c]) + " (area = {:.5f})".format(ap[c]))
                print("[AP] " + str(labels[c]) + ": {:.5f}".format(ap[c]))
        else: # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            n_true = sum(y_true)/len(y_true)
            plt.plot(recall, precision, label = "AP = {:.5f}".format(ap))
            plt.plot([0, 1], [n_true, n_true], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        if len(labels) > 0: plt.legend(loc="best")
        plt.title("Precision-recall curve")
        pdf.savefig()
        plt.close()

        # Loss
        if len(loss) > 0:
            max_epochs = max([len(l) for k, l in loss.items()])
            for k, l in loss.items():
                plt.plot(np.arange(max_epochs), l, label = k)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.xlim([0, max_epochs])
            plt.legend(loc="best")
            plt.title("Training Loss")
            pdf.savefig()
            plt.close()

        # F1 score
        f1 = []
        if len(labels) > 0: # Multiclass classification
            if not multilabel:
                y_true = torch.argmax(y_true, axis = 1)
                y_pred = torch.argmax(torch.tensor(y_pred), axis = 1)
            else: y_pred = (y_pred > 0.5)
            f1 = f1_score(y_true, y_pred, range(len(labels)), average = None)
            for c in range(len(f1)):
                print("[F1] " + str(labels[c]) + ": {:.5f}".format(f1[c]))
        return roc, ap, f1


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
