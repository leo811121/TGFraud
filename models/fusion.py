import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation
from models.transformer_encoder import Transformer_encoder

class Fusion(nn.Module):
    def __init__(self,):
        super(Fusion, self).__init__()
        self.Linear_w = nn.Sequential(
                    nn.Linear(16, 1),
                    #nn.ReLU(),
                    #nn.Tanh()
                    )
        self.Linear_fusion = nn.Sequential(
                    nn.Linear(16*3, 16*2),
                    nn.ReLU(True),
                    #nn.Tanh()
                    )
        self.Linear_fusion_2 = nn.Sequential(
                    nn.Linear(16*2, 16),
                    nn.ReLU(True),
                    #nn.Tanh()
                    )
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_embd_burst, x_embd_trans, x_burst_trans):
        #W_burst = self.Linear_w(x_embd_burst)
        #W_trans = self.Linear_w(x_embd_trans)
        #W_burst_trans = self.Linear_w(x_burst_trans)
        #W_all = self.Softmax(torch.cat([W_burst, W_trans, W_burst_trans], dim=-1))
        #x_cat = torch.cat([W_all[:,0].unsqueeze(-1)*x_embd_burst, W_all[:,1].unsqueeze(-1)*x_embd_trans, W_all[:,2].unsqueeze(-1)*x_burst_trans], dim=-1)
        x_cat = torch.cat([x_embd_burst, x_embd_trans, x_burst_trans], dim=-1)
        x_lin = self.Linear_fusion(x_cat)
        output = self.Linear_fusion_2(x_lin)
        return output

'''
class Fusion(nn.Module):
    def __init__(self,):
        super(Fusion, self).__init__()
        self.Linear_fusion = nn.Sequential(
                    nn.Linear(16, 1),
                    nn.ReLU(True),
                    #nn.Tanh()
                    )
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_embd_burst, x_embd_trans, x_burst_trans):
        W_burst = self.Linear_fusion(x_embd_burst)
        W_trans = self.Linear_fusion(x_embd_trans)
        W_burst_trans = self.Linear_fusion(x_burst_trans)
        W_all = torch.cat([W_burst, W_trans, W_burst_trans], dim=-1)
        #print("W_all", W_all.size())
        W_all = self.Softmax(W_all)
        #print("W_all[:,0]", W_all[:,0].size())
        output = W_all[:,0].unsqueeze(-1)*x_embd_burst + W_all[:,1].unsqueeze(-1)*x_embd_trans + W_all[:,2].unsqueeze(-1)*x_burst_trans
        #output = torch.cat([W_all[:,0].unsqueeze(-1)*x_embd_burst, W_all[:,1].unsqueeze(-1)*x_embd_trans], dim=-1)
        return output
'''   

