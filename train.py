import time
import argparse
import sys
import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
from training.gnn_training import train_gnn_y
from utils.utils import auto_select_gpu


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='alpha',)
parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE_EGSAGE')
parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
parser.add_argument('--concat_states', action='store_true', default=False)
parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
parser.add_argument('--inductive_rate', type=float, default=1.0)
parser.add_argument('--aggr', type=str, default='max',)
parser.add_argument('--node_dim', type=int, default=16)
parser.add_argument('--edge_dim', type=int, default=8)
parser.add_argument('--neigh_num', type=int, default=10)
parser.add_argument('--edge_trans_dim', type=int, default=8)
parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight 1: as input to mlp 
parser.add_argument('--gnn_activation', type=str, default='relu')
parser.add_argument('--impute_hiddens', type=str, default='')
parser.add_argument('--impute_activation', type=str, default='relu')
parser.add_argument('--predict_hiddens', type=str, default='')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--kfold', type=int, default=10)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_restart', type=int, default=0)
parser.add_argument('--opt_decay_step', type=int, default=2)
parser.add_argument('--opt_decay_rate', type=float, default=0.8)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    print('Using CPU')
    device = torch.device('cpu')

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)


train_gnn_y(args, device)