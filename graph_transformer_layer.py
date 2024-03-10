import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def src_dot_dst_bias(src_field, dst_field, out_field, att_bais):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True) + edges.dst[att_bais]}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, att_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.att_bias = att_bias
        if self.att_bias:
            self.att_bias_layer = torch.nn.Linear(1, num_heads, True)  # torch.nn.Embedding(10,num_heads)

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        if self.att_bias:
            g.ndata['attention_bias'] = self.att_bias_layer(g.ndata['dis'].reshape(-1, 1).float()).reshape(-1,
                                                                                                           self.num_heads,
                                                                                                           1)
            g.apply_edges(src_dot_dst_bias('K_h', 'Q_h', 'score', 'attention_bias'))  # , edges)
            # g.ndata.pop('attention_bias')
        else:
            g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, source_node_type=None, target_node_type=None):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        if source_node_type is None:
            g_node_view = g.ndata
        else:
            g_node_view = g.nodes[source_node_type].data
        g_node_view['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g_node_view['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g_node_view['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        if target_node_type is None:
            g_target_node_view = g.ndata
        else:
            g_target_node_view = g.nodes[target_node_type].data
        head_out = g_target_node_view['wV'] / g_target_node_view['z']

        return head_out


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False, att_bias=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias, att_bias)

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in1 = h  # for first residual guided

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)

        if self.residual:
            h = h_in1 + h  # residual guided

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual guided

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual guided

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
