import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric
import math

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_activation

class Encoder(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, head=1):
        super(Encoder, self).__init__(aggr='mean')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head = head
        self.tanh = nn.Tanh()
        
        self.lin_value = nn.Sequential(
            nn.Linear(in_channels, head * out_channels),
            nn.ReLU(True)
        )
        self.lin_timeDiff = nn.Sequential(
            nn.Linear(1, head * out_channels, bias=False),
            nn.ReLU(True)
        )
        self.lin_aggrOut = nn.Sequential(
            nn.Linear(out_channels*2, head * out_channels),
            nn.Tanh()
        )


    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        w_timeDiff = edge_attr[:,0]
        inf_num = torch.ones_like(w_timeDiff) * (10**-10)
        w_timeDiff = torch.where(w_timeDiff == 0, inf_num, w_timeDiff)
        w_timeDiff_inv = (w_timeDiff**-1).unsqueeze(-1)
        w_timeDiff_inv = torch_geometric.utils.softmax(self.tanh(w_timeDiff_inv), edge_index[0])
        w_timeDiff_inv_lin = self.lin_timeDiff(w_timeDiff_inv)
        value = self.lin_value(edge_attr[:,1:]).view(-1, self.out_channels)
        m_j = torch.cat([w_timeDiff_inv_lin, value], dim=-1)
        return m_j
   
    def update(self, aggr_out, x):
        aggr_out = self.lin_aggrOut(aggr_out)
        aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
    
    
   
    
    
    
