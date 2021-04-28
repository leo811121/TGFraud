import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation

class MLPNet(torch.nn.Module):
    def __init__(self, 
         		input_dims, output_dim,
         		hidden_activation='relu',
         		output_activation=None,
                dropout=0.):
        super(MLPNet, self).__init__()

        layers = nn.ModuleList()

        layer = nn.Sequential(
        				nn.Linear(input_dims, 16),
                #nn.Sigmoid(),
                #nn.ReLU(),
                #get_activation(output_activation),
                #nn.Softmax()
        				)
       	layers.append(layer)
            
        layer = nn.Sequential(
        				nn.Linear(16, 16),
                #nn.Sigmoid(),
                #nn.ReLU(),
                #get_activation(output_activation),
                #nn.Softmax()
        				)
        layers.append(layer)
        
        layer = nn.Sequential(
                nn.Linear(16, 2),
                #nn.Sigmoid(),
                #nn.ReLU(),
                #get_activation(output_activation),
                #nn.Softmax()
                )
        layers.append(layer)
        
       	self.layers = layers
      
    def forward(self, inputs):
        input_var = inputs
        for layer in self.layers:
          input_var = layer(input_var)
        return input_var




