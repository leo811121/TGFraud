import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation

class normalize(nn.Module):
    def __init__(self, input_shape):
        super(normalize, self).__init__()
        self.inputs_shape = input_shape
        self.instaNorm = nn.InstanceNorm1d(input_shape)
        self.params_shape = input_shape
        self.beta = nn.Parameter(torch.zeros(self.params_shape))
        self.gamma = nn.Parameter(torch.ones(self.params_shape))
        
    def forward(self, feature_input):
        self.normalized = self.instaNorm(feature_input)
        self.outputs = self.gamma * self.normalized + self.beta
        return self.outputs

class Transformer_encoder(nn.Module):
    def __init__(self, feature_input_shape, num_units, edge_num, num_heads=1, has_residual=True, dropout=1.0):
        super(Transformer_encoder, self).__init__() 
        self.num_heads = num_heads
        self.has_residual = has_residual
        self.Norm_time = nn.InstanceNorm1d(edge_num)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.w_time_inv = nn.Linear(1, num_units, bias=False) 
        self.w_time = nn.Linear(1, num_units, bias=False)
        #self.smoothing = GaussianSmoothing(1, 5, 1)
        self.w_Q = nn.Sequential(nn.Linear(feature_input_shape, num_units, bias=False), 
                      #nn.Tanh()
                      nn.ReLU(True)
                      )
        self.w_K = nn.Sequential(nn.Linear(feature_input_shape, num_units, bias=False),
                      nn.ReLU(True)
                      )
        self.w_V = nn.Sequential(nn.Linear(feature_input_shape, num_units, bias=False),
                      nn.ReLU(True)
                      )
        self.w_V_res = nn.Sequential(nn.Linear(feature_input_shape, num_units, bias=False),
                      nn.ReLU(True)
                      )
        self.Relu = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.normalize = normalize(num_units)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, time_diff):
        mask_timeNorm = torch.sign(torch.abs(time_diff))
        time_Diff_inv = torch.where(mask_timeNorm == 0, mask_timeNorm , time_diff**-1)
        timeDiff_Norm = self.tanh(time_Diff_inv)
        timdDiff_Lin = self.w_time_inv(timeDiff_Norm)
        value = self.w_time(time_diff)
        
        self.Q = self.w_Q(timdDiff_Lin)
        self.Q = torch.cat(torch.split(self.Q, int(self.Q.size()[-1]/self.num_heads), dim=2), dim=0)
        self.Q = mask_timeNorm * self.Q
        self.K = self.w_K(timdDiff_Lin)
        self.K = torch.cat(torch.split(self.K, int(self.K.size()[-1]/self.num_heads), dim=2), dim=0)
        self.K = mask_timeNorm * self.K
        self.V = self.w_V(value)
        self.V = torch.cat(torch.split(self.V, int(self.V.size()[-1]/self.num_heads), dim=2), dim=0)
        self.V = mask_timeNorm * self.V

        if self.has_residual:
            self.V_res = self.w_V_res(value)
            self.V_res = mask_timeNorm * self.V_res
            
        scaledSimilary = torch.matmul(self.Q, self.K.permute(0,2,1)) / (self.K.size()[-1]**0.5)
        keymasks = torch.sign(torch.abs(torch.sum(scaledSimilary, dim=-1)))
        weights = self.softmax(scaledSimilary)


        self.outputs = torch.matmul(weights, self.V)
        self.outputs = torch.cat(torch.split(self.outputs, int(self.outputs.size()[0]/self.num_heads), dim=0), dim=2)
        self.outputs = self.dropout(self.outputs)
        if self.has_residual:
            self.outputs += self.V_res
        self.outputs = self.Relu(self.outputs)
        self.outputs = self.normalize(self.outputs)
        
        return self.outputs, scaledSimilary