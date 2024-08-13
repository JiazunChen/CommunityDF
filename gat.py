# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import SumPooling, AvgPooling

from graph_transformer_layer import GraphTransformerLayer


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def make_linear_block(in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=False, dropout=0.):
    return LinearBlock(in_size, out_size, act_cls, norm_type, bias, residual, dropout)


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
        super().__init__()
        self.residual = residual and (in_size == out_size)
        layers = []
        if norm_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(in_size))
        elif norm_type == 'layer_norm':
            layers.append(nn.LayerNorm(in_size))
        elif norm_type is not None:
            raise NotImplementedError
        layers.append(nn.Linear(in_size, out_size, bias))
        if act_cls is not None:
            layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        z = self.f(x)
        if self.residual:
            z += x
        return z


class GAT(nn.Module):
    def __init__(self, input, output, heads, dropout, concatenate=True):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.concatenate = concatenate
        # two-layer GAT
        for head in heads[:-1]:
            self.gat_layers.append(
                dglnn.GATConv(
                    input,
                    output // head,
                    head,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            dglnn.GATConv(
                output,
                output // heads[-1],
                heads[-1],
                feat_drop=dropout,
                attn_drop=dropout,
                activation=None,
            )
        )

    def forward(self, batch_graph, x):
        hs = [x]
        for i, layer in enumerate(self.gat_layers):
            h = layer(batch_graph, hs[-1])

            hs.append(h.flatten(1))

        if self.concatenate:
            return torch.concatenate(hs, dim=1)
        else:
            return hs[-1].flatten(1)


class GraphTransformer(nn.Module):
    def __init__(self, input, output, heads, dropout, concatenate=False, residual=False, att_bias=False):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.concatenate = concatenate
        self.residual = residual
        self.att_bias = att_bias

        # two-layer GAT
        for head in heads[:-1]:
            self.gat_layers.append(
                GraphTransformerLayer(input,
                                      input,
                                      head, dropout, att_bias=att_bias)
            )
            '''
             dglnn.GATConv(
                input,
                output//head,
                head,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
            )
            
                dglnn.GATConv(
                    output,
                    output//heads[-1],
                    heads[-1],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
            '''
        self.gat_layers.append(
            GraphTransformerLayer(input,
                                  output,
                                  heads[-1], dropout, att_bias=att_bias)
        )

    def forward(self, batch_graph, x):
        hs = [x]
        for i, layer in enumerate(self.gat_layers):
            h = layer(batch_graph, hs[-1])
            if self.residual:
                h += x
            hs.append(h.flatten(1))
            # print(h.shape, hs[-1].shape)
            # print(hs[-1].shape)
        if self.concatenate:
            return torch.concatenate(hs, dim=1)
        else:
            return hs[-1].flatten(1)


class SelfAttnPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, batch_graph, x):
        batched_graphs = dgl.unbatch(batch_graph)
        pooled_emds = []
        idx = 0
        for subgraph in batched_graphs:
            subgraph_x = x[idx:idx + len(subgraph.nodes())]
            # print(idx,idx+len(subgraph.nodes()))
            w = self.score_layer(subgraph_x)
            w = torch.softmax(w, dim=0)
            pooled_emd = torch.sum(subgraph_x * w, dim=0, keepdim=True)
            pooled_emds.append(pooled_emd)
            idx += len(subgraph.nodes())
        pooled_emds = torch.cat(pooled_emds, dim=0)
        return pooled_emds


class PredictComsize(nn.Module):
    def __init__(self, args, input, output, dropout, device, num_classes=1):
        super().__init__()
        self.args = args
        self.input = input
        self.output = output
        self.device = device
        self.num_classes = num_classes
        self.seed_embedding = nn.Linear(1, input, bias=False)
        self.node_embedding = nn.Linear(1, input, bias=False)
        self.degree_embedding = nn.Linear(1, input, bias=False)
        self.cluster_embedding = nn.Linear(1, input, bias=False)
        if args.feat_dim != 0:
            self.feat_embedding = nn.Linear(args.feat_dim, input, bias=False)
        if args.pos_enc_dim is not None:
            self.pos_embedding = nn.Linear(args.pos_enc_dim, input, bias=False)
        self.gat = GraphTransformer(input, output, [4, 4, 4], dropout)

        mlpinput = output

        self.mlp = nn.Sequential(
            make_linear_block(mlpinput, output, nn.ReLU, None, dropout=dropout),
            make_linear_block(output, num_classes, None, None, dropout=dropout),
        )

    def forward(self, graph, X, Seed, D, C):
        graph = graph.to(self.device)
        newnodeemd = self.node_embedding(X.to(self.device))
        Seed = self.seed_embedding(Seed.to(self.device))
        D = self.degree_embedding(D.to(self.device))
        C = self.cluster_embedding(C.to(self.device))
        input_X = X + Seed + D + C
        if self.args.pos_enc_dim:
            Pos = self.pos_embedding(graph.ndata['lap_pe'])
            input_X += Pos
        if self.args.feat_dim != 0:
            nodefeat = graph.ndata['feat'].float()
            nodefeat = self.feat_embedding(nodefeat.to(self.device))
            # print(nodefeat[:3][:3])
            input_X += nodefeat
        nodeemd = self.gat(graph, input_X)
        newnodeemd = nodeemd
        predX = self.mlp(newnodeemd)
        return predX

from dgl.nn import GraphConv
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes,dropout = 0.0):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.classify =  nn.Sequential(
            make_linear_block(hidden_feats, hidden_feats, nn.ReLU, None, dropout=dropout),
            make_linear_block(hidden_feats, num_classes, None, None, dropout=dropout),
        )
        self.sumpool = SumPooling()
        self.num_classes = num_classes

    def forward(self, g, features,seeds):
        x = F.relu(self.conv1(g, features))
        x = F.relu(self.conv2(g, x))
        if self.num_classes == 1:
            #x = self.sumpool(g, x)
            x = x[seeds]
        return self.classify(x)
