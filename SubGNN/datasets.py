# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Typing
from typing import List

class SubgraphDataset(Dataset):
    '''
    Stores subgraphs and their associated labels as well as precomputed similarities and border sets for the subgraphs
    '''

    def __init__(self, subgraph_list: List, labels, cc_ids, N_border, NP_sim, I_S_sim, B_S_sim, multilabel, multilabel_binarizer):
        # subgraph ids & labels
        self.subgraph_list = subgraph_list
        self.cc_ids = cc_ids
        self.labels = labels
        
        # precomputed border set
        self.N_border = N_border

        # precomputed similarity matrices
        self.NP_sim = NP_sim
        self.I_S_sim = I_S_sim
        self.B_S_sim = B_S_sim

        # necessary for handling multi-label classsification
        self.multilabel = multilabel
        self.multilabel_binarizer = multilabel_binarizer

    def __len__(self):
        '''
        Returns number of subgraphs
        '''
        return len(self.subgraph_list)

    def __getitem__(self, idx):
        '''
        Returns a single example from the datasest
        '''
        
        subgraph_ids = torch.LongTensor(self.subgraph_list[idx]) # list of node IDs in subgraph

        cc_ids = self.cc_ids[idx]
        N_border = self.N_border[idx] if self.N_border != None else None
        NP_sim = self.NP_sim[idx] if self.NP_sim != None else None
        I_S_sim = self.I_S_sim[idx] if self.I_S_sim != None else None
        B_S_sim = self.B_S_sim[idx] if self.B_S_sim != None else None
        
        if self.multilabel:
            label = torch.LongTensor(self.multilabel_binarizer.transform([self.labels[idx]]))
        else:
            label = torch.LongTensor([self.labels[idx]])
        idx = torch.LongTensor([idx])

        return (subgraph_ids, cc_ids, N_border, NP_sim, I_S_sim, B_S_sim, idx, label)
