import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric
from torch.nn.init import xavier_uniform_, zeros_
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_activation

class EGraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 edge_channels, activation, edge_mode,
                 normalize_emb,
                 aggr):
        super(EGraphSage, self).__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode
        self.tanh = nn.Tanh()
        if edge_mode == 0:
            self.message_lin = nn.Linear(in_channels, out_channels)
            self.attention_lin = nn.Linear(2*in_channels+edge_channels, 1)
        elif edge_mode == 1:
            self.message_lin = nn.Linear(in_channels+edge_channels, out_channels)
        elif edge_mode == 2:
            self.message_lin = nn.Linear(2*in_channels+edge_channels, out_channels)
        elif edge_mode == 3:
            self.message_lin = nn.Sequential(
                    nn.Linear(2*in_channels+edge_channels, out_channels),
                    get_activation(activation),
                    nn.Linear(out_channels, out_channels),
                    )
        elif edge_mode == 4:
            self.message_lin = nn.Linear(in_channels, out_channels*edge_channels)
        elif edge_mode == 5:
            self.message_lin = nn.Linear(2*in_channels, out_channels*edge_channels)
        elif edge_mode == 6:
            self.source_lin = nn.Linear(in_channels, 8)
            self.message_lin = nn.Linear(8+edge_channels, out_channels)
        elif edge_mode == 7:
            self.message_lin = nn.Linear(edge_channels, out_channels)
            

        self.agg_lin = nn.Linear(in_channels+out_channels, out_channels)

        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)
        self.normalize_emb = normalize_emb

        ##GGNN
        self.GRU = nn.GRU(out_channels, in_channels, 1)
        self.Lin_GRU = nn.Linear(in_channels, out_channels)

        ##ResNET
        self.lin_res = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Sigmoid()
        )

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 1:
            m_j = torch.cat((x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 2 or self.edge_mode == 3:
            m_j = torch.cat((x_i,x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 4:
            E = x_j.shape[0]
            w = self.message_lin(x_j)
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 5:
            E = x_j.shape[0]
            w = self.message_lin(torch.cat((x_i,x_j),dim=-1))
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 6:
            m_j = torch.cat((self.source_lin(x_j), edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 7:
            m_j = edge_attr
            m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        aggr_out = aggr_out.unsqueeze(0)
        x = x.unsqueeze(0)
        aggr_out, _ = self.GRU(aggr_out, x)
        aggr_out = self.Lin_GRU(aggr_out.squeeze(0))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
    
    
