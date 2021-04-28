import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation
from models.transformer_encoder import Transformer_encoder

class Encoder(nn.Module):
    def __init__(self, feature_input_shape, num_units, out_channel, max_edge_count, edge_num=200, num_heads=2, has_residual=True, dropout=1.0):
        super(Encoder, self).__init__()
        self.Transformer_encoder = Transformer_encoder(feature_input_shape, num_units, edge_num)
        #self.Transformer_encoder_2 = Transformer_encoder(16, 16, edge_num)
        self.Linear_trans = nn.Sequential(
        				nn.Linear(max_edge_count, out_channel),
                nn.Sigmoid()
        				)
        

    def forward(self, time_diff):
        output, weights = self.Transformer_encoder(time_diff)
        #output,_ = self.Transformer_encoder_2(output, output, output, time_diff)
        #output = torch.max(output, dim=-1)[0]
        output = torch.mean(output, dim=-1)
        output = self.Linear_trans(output)
        return output, weights