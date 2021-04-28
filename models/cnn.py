import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation

class Cnn(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Cnn, self).__init__()
        self.Conv2d = nn.Conv2d(1, 1, 2, stride=1)
        self.Lin_burst = nn.Linear(input_dims*2, input_dims)
        self.Lin_calm = nn.Linear(input_dims*2, input_dims)
   
    def forward(self, x_trans_burst, x_trans_calm, x_embd):
        #print("x_trans_burst", x_trans_burst.size())
        #print("x_trans_calm", x_trans_calm.size())
        x_embd = x_embd.unsqueeze(1).repeat(1,x_trans_calm.size(1),1)
        #print("x_embd", x_embd.size())
        x_cat_burst = torch.cat([x_embd, x_trans_burst], dim=-1)
        #print("x_cat_burst", x_cat_burst.size())
        x_cat_calm = torch.cat([x_embd, x_trans_calm], dim=-1)
        #print("x_cat_calm", x_cat_calm.size())
        x_burst_calm = torch.matmul(x_cat_burst, x_cat_calm.permute(0,2,1))
        #print("x_burst_calm", x_burst_calm.size())
        x_burst_calm = x_burst_calm.unsqueeze(1)
        #print("x_burst_calm", x_burst_calm.size())
        x_burst_calm = self.Conv2d(x_burst_calm).view(x_trans_calm.size(0), -1)
        return x_burst_calm
        
        
        
