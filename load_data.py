import numpy as np
import torch
import random
import networkx as nx
import collections
import pathlib
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


def preprocess_nodefeats(nodefeats):
    ind = np.array([[i, j] for i, js in nodefeats.items() for j in js])
    sp_feats = sp.csr_matrix((np.ones(len(ind)), (ind[:, 0], ind[:, 1])))
    svd = TruncatedSVD(n_components=200, algorithm='arpack')
    x = svd.fit_transform(sp_feats)
    x = (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True)
    return x


def load_dataset(root, name):
    root = pathlib.Path(root)
    prefix = f'{name}-1.90'
    print(root)
    with open(root / f'{prefix}.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')
        edges = np.array([[int(i) for i in x.split()] for x in edges])
    with open(root / f'{prefix}.cmty.txt') as fh:
        comms = fh.read().strip().split('\n')
        comms = [[int(i) for i in x.split()] for x in comms]
    g = nx.Graph()
    g.add_edges_from(edges)
    if (root / f'{prefix}.nodefeat.txt').exists():
        with open(root / f'{prefix}.nodefeat.txt') as fh:
            nodefeats = [x.split() for x in fh.read().strip().split('\n')]
            nodefeats = {int(k): [int(i) for i in v] for k, *v in nodefeats}
            # nodefeats = preprocess_nodefeats(nodefeats)
            # print('with feature')
            if name == 'twitter':
                feature_matrix = preprocess_nodefeats(nodefeats)
            else:
                max_feature_index = max(max(features) for features in nodefeats.values() if len(features))
                num_features = max_feature_index + 1  # 索引从0开始，所以加1
                feature_matrix = np.zeros((len(nodefeats), num_features))
                for node, features in nodefeats.items():
                    feature_matrix[node, features] = 1



    else:
        feature_matrix = None  # np.random.randn(nx.number_of_nodes(g),100)
    # graph = Graph(edges)

    return g, comms, feature_matrix, prefix


def split_comms(graph, comms, train_size):
    train_comms, valid_comms = comms[:train_size], comms[train_size:]
    test_comms = []
    print(f'blen {len(train_comms)} {len(valid_comms)}')
    train_comms = [list(x) for nodes in train_comms for x in connected_components(graph, nodes) if len(x) >= 3]
    valid_comms = [list(x) for nodes in valid_comms for x in connected_components(graph, nodes) if len(x) >= 3]
    print(f'alen {len(train_comms)} {len(valid_comms)}')
    max_size = max(len(x) for x in train_comms + valid_comms + test_comms)
    return train_comms, valid_comms, test_comms, max_size


def load_data(root='datasets', dataset='facebook', train_size=10):
    # args = self.args
    graph, comms, nodefeats, ds_name = load_dataset(root, dataset)
    train_comms, valid_comms, test_comms, max_size = split_comms(graph, comms, train_size)
    print(f'[{ds_name}] # Nodes: {nx.number_of_nodes(graph)} Edges: {nx.number_of_edges(graph)} ', flush=True)
    print(f'[# comms] Train: {len(train_comms)} Valid: {len(valid_comms)} Test: {len(test_comms)}', flush=True)
    return graph, train_comms, valid_comms, nodefeats


def local_degree_profile(graph, seed, maper):
    n_nodes = graph.number_of_nodes()
    feat_mat = np.zeros([n_nodes, 8], dtype=np.float32)
    neighbors = {}
    for node in graph.nodes:
        neighbors[node] = set([maper[u] for u in graph.neighbors(node)])
        feat_mat[maper[node]][0] = len(neighbors[node])
        neighbor_degs = feat_mat[list(neighbors[node]), 0]
        feat_mat[maper[node], 1:5] = neighbor_degs.min(), neighbor_degs.max(), neighbor_degs.mean(), neighbor_degs.std()
    triangle = {i: 0 for i in range(n_nodes)}
    for u, v in graph.edges:
        intersection = neighbors[u] & neighbors[v]
        if len(intersection) != 0:
            triangle[maper[u]] += len(intersection)
            triangle[maper[v]] += len(intersection)
    triangle = {i: triangle[i] // 2 for i in range(n_nodes)}
    for i in range(n_nodes):
        feat_mat[i, 5] = triangle[i]
    feat_mat = (feat_mat - feat_mat.mean(0, keepdims=True)) / (feat_mat.std(0, keepdims=True) + 1e-9)
    feat_mat[:, 6] = 1
    feat_mat[maper[seed], 7] = 1
    return feat_mat


def connected_components(g, nodes):  # 旧社区在新图中可能被拆成几个新社区

    remaining = set(nodes)
    ccs = []
    cc = set()
    queue = collections.deque()
    while len(remaining) or len(queue):
        # print(queue, remaining)
        if len(queue) == 0:
            if len(cc):
                ccs.append(cc)
            v = remaining.pop()
            cc = {v}
            queue.extend(set(g.neighbors(v)) & remaining)
            remaining -= {v}
            remaining -= set(g.neighbors(v))
        else:
            v = queue.popleft()
            queue.extend(set(g.neighbors(v)) & remaining)
            cc |= (set(g.neighbors(v)) & remaining) | {v}
            remaining -= set(g.neighbors(v))
    if len(cc):
        ccs.append(cc)

    # print(nodes,ccs)
    return ccs


def pre_subgraph(graph, train_coms, features, length=400):
    com_nodes = set([node for com in train_coms for node in com])
    sg_nodes = {}
    sg_edges = {}
    sg_train_nodes = {}
    sg_test_nodes = {}
    sg_features = {}
    sg_targets = {}
    sg_localfeat = {}
    for k in range(len(train_coms)):
        # k = 0
        com = train_coms[k]
        seed = np.random.choice(com, size=1)[0]
        allNodes = []
        allNodes.append(seed)
        vis = {}
        vis[seed] = 1
        pos = 0
        while pos < len(allNodes) and pos < length and len(allNodes) < length:
            cnode = allNodes[pos]
            for nb in graph.neighbors(cnode):
                if nb not in vis.keys() and len(allNodes) < length:
                    allNodes.append(nb)
                    vis[nb] = 1
            pos = pos + 1
        miss = 0
        posNodes = []
        for node in com:
            if node not in allNodes:
                miss += 1
            else:
                posNodes.append(node)
        negNodes = list(set(allNodes) - set(posNodes))
        random.shuffle(negNodes)
        negNodes = negNodes[:len(posNodes)]
        # print('miss',miss,'all',len(allNodes),'pos',len(posNodes),'neg',len(negNodes))
        allNodes = sorted(allNodes)
        labels = [int(x in com) for x in allNodes]
        subgraph = nx.Graph()
        for i in range(len(allNodes)):
            for j in range(i):
                if ((allNodes[i], allNodes[j]) in graph.edges) or ((allNodes[j], allNodes[i]) in graph.edges):
                    subgraph.add_edge(allNodes[i], allNodes[j])

        # print("size of nodes %d size of edges %d" %(len(subgraph.nodes), len(subgraph.edges)))
        sg_nodes[k] = sorted(list(subgraph.nodes()))
        # 保存概率值
        sg_predProbs = [0.0] * len(sg_nodes[k])
        sg_predLabels = [0] * len(sg_nodes[k])
        # 构建老-》新的映射  和逆向映射
        mapper = {node: i for i, node in enumerate(sg_nodes[k])}
        rmapper = {i: node for i, node in enumerate(sg_nodes[k])}
        # 新的编号
        sg_edges[k] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [
            [mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]

        posNodes = [mapper[node] for node in posNodes]
        negNodes = [mapper[node] for node in negNodes]
        allNodes1 = [mapper[node] for node in allNodes]
        sg_localfeat[k] = local_degree_profile(subgraph, seed, mapper)
        sg_train_nodes[k] = posNodes + negNodes
        sg_train_nodes[k] = sorted(sg_train_nodes[k])
        sg_test_nodes[k] = list(set(allNodes1).difference(set(sg_train_nodes[k])))
        sg_test_nodes[k] = sorted(sg_test_nodes[k])
        sg_features[k] = features[sg_nodes[k], :]
        sg_targets[k] = [0] * len(allNodes1)
        possum = 0
        for i in allNodes1:
            if rmapper[i] in com_nodes and i not in negNodes:
                sg_targets[k][i] = 1
                possum += 1
        if k % 20 == 0:
            print(f'k {k}, possum {possum}, negsum {len(allNodes1) - possum}')
        sg_targets[k] = np.array(sg_targets[k])
        sg_targets[k] = sg_targets[k][:, np.newaxis]
        sg_targets[k] = sg_targets[k].astype(int)

    return sg_nodes, sg_edges, sg_train_nodes, sg_features, sg_targets, sg_test_nodes, sg_localfeat


if __name__ == '__main__':
    load_data()
