import os.path
import dgl
import torch
import numpy as np
import networkx as nx
from getcomm import heap_com_topk
from scipy import sparse as sp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from time import time


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    A = g.adj(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    print(EigVec.shape)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


class Mydata():
    def __init__(self, dataset, graph, subrgaph_size, coms, max_com_size, mode='train',
                 pos_enc_dim=None, shuffle=False, contain_com=False, bfs_pagerank=True, sample_pagerank=None, pool=True,
                 features=None, all_edges=False,
                 cache=True, alpha=0.85, shortest_path=False):
        # ().__init__()
        self.pool = pool
        self.dataset = dataset
        self.graph = graph
        self.features = features
        self.coms = coms
        self.alpha = alpha
        self.subgraph_size = subrgaph_size
        self.n_nodes = self.graph.number_of_nodes()
        self.max_com_size = max_com_size
        self.mode = mode
        self.shuffle = shuffle
        self.pos_enc_dim = pos_enc_dim
        self.contain_com = contain_com
        self.bfs_pagerank = bfs_pagerank
        self.sample_pagerank = sample_pagerank
        self.cache = cache
        self.all_edges = all_edges
        self.shortest_path = shortest_path
        self.filename = f'{dataset}_{subrgaph_size}_{len(self.coms)}_{max_com_size}_{mode}_{shuffle}_{pos_enc_dim}_{contain_com}_{bfs_pagerank}_{sample_pagerank}.pt'
        if self.alpha != 0.85:
            self.filename = f'{self.alpha}_{self.filename}'
        if self.features is not None:
            self.filename = f'True_{self.filename}'
        if self.all_edges:
            self.filename = f'True_{self.filename}'
        if self.shortest_path:
            self.filename = f'True_{self.filename}'
        self.datapath = os.path.join('preprocess', self.filename)
        global myglobalgraph
        myglobalgraph = self.graph
        global nodefeat
        nodefeat = features

        self.load_data()

    def load_data(self):
        if os.path.exists(self.datapath) and self.cache:
            print('load data')
            self.data = torch.load(self.datapath)
        else:
            print('preprocess data')
            self.data = []
            if self.pool:
                po = Pool(cpu_count() - 10)
                print('cpu_count()', cpu_count())
                for i in (range(len(self.coms))):
                    po.apply_async(getdatai_pool, (
                        i, self.coms[i], self.shuffle, self.mode, self.contain_com, self.bfs_pagerank,
                        self.subgraph_size,
                        self.pos_enc_dim,
                        self.sample_pagerank, self.filename, self.alpha, self.shortest_path, self.all_edges))
                po.close()
                po.join()
                for i in (range(len(self.coms))):
                    result = torch.load(os.path.join('preprocess', f'{i}_{self.filename}'))
                    os.remove(os.path.join('preprocess', f'{i}_{self.filename}'))
                    self.data.append(result)
            else:
                for i in tqdm(range(len(self.coms))):
                    self.data.append(self.getdatai(i))
            torch.save(self.data, self.datapath)

    def __len__(self):
        return len(self.coms)

    def __getitem__(self, k):
        return self.data[k]

    def getdatai(self, k):
        # print('k',k)
        st = time()
        com = self.coms[k]
        if self.shuffle:
            np.random.shuffle(com)
        if self.mode == 'train' and self.shuffle:
            seed = com[0]
        else:
            seed = max(com)
        if self.contain_com:
            # allNodes=[seed]
            allNodes = com[:]
        else:
            allNodes = [seed]
        # print('len',len(com))
        if self.bfs_pagerank and not self.contain_com:
            probx = nx.pagerank(self.graph, personalization={seed: 1}, alpha=self.alpha, max_iter=1000)
            allNodes = heap_com_topk(seed, self.graph, probx, self.subgraph_size)
        else:
            vis = {node: 1 for node in allNodes}
            pos = 0
            while pos < len(allNodes) and pos < self.subgraph_size and len(allNodes) < self.subgraph_size:
                cnode = allNodes[pos]
                for nb in self.graph.neighbors(cnode):
                    if nb not in vis.keys() and len(allNodes) < self.subgraph_size:
                        allNodes.append(nb)
                        vis[nb] = 1
                pos = pos + 1

        allNodes = sorted(allNodes)
        subgraph = nx.subgraph(self.graph, allNodes)
        sg_nodes = sorted(list(subgraph.nodes()))
        assert sg_nodes == allNodes
        sg_edges = list(subgraph.edges)

        mapper = {node: i for i, node in enumerate(sg_nodes)}
        rmapper = {i: node for i, node in enumerate(sg_nodes)}
        # 新的编号
        new_sg_edges = [[mapper[edge[0]], mapper[edge[1]]] for edge in sg_edges] + [
            [mapper[edge[1]], mapper[edge[0]]] for edge in sg_edges]
        sg = nx.Graph()
        sg.add_edges_from(new_sg_edges)

        labels = [int(x in com) for x in allNodes]
        seed_labels = [int(x == seed) for x in allNodes]
        assert sum(seed_labels) == 1
        newcom = [rmapper[i] for i in range(len(labels)) if labels[i] == 1]
        miss = len(com) - sum(labels)
        # print(sorted(newcom),'\n',sorted(com))
        assert set(newcom).issubset(set(com))

        new_sg_edges_T = torch.tensor(new_sg_edges).T

        dgl_graph = dgl.add_self_loop(dgl.graph((new_sg_edges_T[0], new_sg_edges_T[1])))
        if self.pos_enc_dim is not None:
            h, ev = dgl.lap_pe(dgl_graph, self.pos_enc_dim, padding=True, return_eigval=True)
            # print(h,ev)
            dgl_graph.ndata['lap_pe'] = h
        if self.sample_pagerank is not None:
            probx = nx.pagerank(sg, personalization={mapper[seed]: 1}, alpha=self.alpha, max_iter=1000)
            pg = [item[1] for item in sorted(list(probx.items()))]
            dgl_graph.ndata['pg'] = torch.tensor(pg)
            if self.sample_pagerank == 'normal':
                dgl_graph.ndata['pg'] = (dgl_graph.ndata['pg'] - torch.mean(dgl_graph.ndata['pg'])) / torch.std(
                    dgl_graph.ndata['pg'])
                # print(dgl_graph.ndata['pg'])
            if self.sample_pagerank == 'minmax':
                dgl_graph.ndata['pg'] = dgl_graph.ndata['pg'] / torch.max(dgl_graph.ndata['pg'])
                # print(dgl_graph.ndata['pg'])
        dgl_graph.ndata['x'] = torch.tensor([int(rmapper[int(x)] in com) for x in dgl_graph.nodes()])
        dgl_graph.ndata['seed'] = torch.tensor([int(rmapper[int(x)] == seed) for x in dgl_graph.nodes()])
        triangles = nx.triangles(sg)
        dgl_graph.ndata['triangles'] = torch.tensor([triangles[int(x)] for x in dgl_graph.nodes()])
        clustering_coefficients = nx.clustering(sg)
        dgl_graph.ndata['clustering_coefficients'] = torch.tensor(
            [clustering_coefficients[int(x)] for x in dgl_graph.nodes()]).float()
        if self.features is not None:
            dgl_graph.ndata['feat'] = torch.tensor([self.features[rmapper[int(x)]] for x in dgl_graph.nodes()])
        if self.shortest_path:
            dis = nx.shortest_path(sg, mapper[seed])
            dgl_graph.ndata['dis'] = torch.tensor([len(dis[int(x)]) - 1 for x in dgl_graph.nodes()])
        if self.pool:
            torch.save({'dgl_graph': dgl_graph, 'rmapper': rmapper, 'labels': labels, 'seed': mapper[seed], 'com': com,
                        'miss': miss, 'sg_size': len(sg_nodes), 'seed_labels': seed_labels, 'sg': sg},
                       os.path.join('preprocess', f'{k}_{self.filename}'))
            print(k, "执行完毕，耗时%0.2f" % (time() - st))
        else:
            return {'dgl_graph': dgl_graph, 'rmapper': rmapper, 'labels': labels, 'seed': mapper[seed], 'com': com,
                    'miss': miss, 'sg_size': len(sg_nodes), 'seed_labels': seed_labels, 'sg': sg}


def getdatai_pool(k, com, shuffle, mode, contain_com, bfs_pagerank, subgraph_size, pos_enc_dim, sample_pagerank,
                  filename, alpha, shortest_path, all_edges):
    st = time()
    if shuffle:
        np.random.shuffle(com)
    if mode == 'train' and shuffle:
        seed = com[0]
    else:
        seed = max(com)
    if contain_com:
        allNodes = com[:]
    else:
        allNodes = [seed]
    if bfs_pagerank and not contain_com:
        # new tmp
        if myglobalgraph.number_of_nodes() > 5000:
            vis = {node: 1 for node in allNodes}
            pos = 0
            tmpallNodes = [seed]
            while pos < len(tmpallNodes) and pos < 1000 and len(tmpallNodes) < 1000:
                cnode = tmpallNodes[pos]
                for nb in myglobalgraph.neighbors(cnode):
                    if nb not in vis.keys() and len(tmpallNodes) < subgraph_size:
                        tmpallNodes.append(nb)
                        vis[nb] = 1
                pos = pos + 1
            tmpsubgraph = nx.subgraph(myglobalgraph, tmpallNodes)
            tmpedges = tmpsubgraph.edges()
            tmpsubgraph = nx.Graph()
            tmpsubgraph.add_edges_from(tmpedges)
            probx = nx.pagerank(tmpsubgraph, personalization={seed: 1}, alpha=alpha, max_iter=1000)
            allNodes = heap_com_topk(seed, tmpsubgraph, probx, subgraph_size)
        else:
            probx = nx.pagerank(myglobalgraph, personalization={seed: 1}, alpha=alpha, max_iter=1000)
            allNodes = heap_com_topk(seed, myglobalgraph, probx, subgraph_size)
    else:
        vis = {node: 1 for node in allNodes}
        pos = 0
        while pos < len(allNodes) and pos < subgraph_size and len(allNodes) < subgraph_size:
            cnode = allNodes[pos]
            for nb in myglobalgraph.neighbors(cnode):
                if nb not in vis.keys() and len(allNodes) < subgraph_size:
                    allNodes.append(nb)
                    vis[nb] = 1
            pos = pos + 1

    allNodes = sorted(allNodes)
    subgraph = nx.subgraph(myglobalgraph, allNodes)
    sg_nodes = sorted(list(subgraph.nodes()))
    assert sg_nodes == allNodes
    sg_edges = list(subgraph.edges)

    mapper = {node: i for i, node in enumerate(sg_nodes)}
    rmapper = {i: node for i, node in enumerate(sg_nodes)}
    # 新的编号
    new_sg_edges = [[mapper[edge[0]], mapper[edge[1]]] for edge in sg_edges] + [
        [mapper[edge[1]], mapper[edge[0]]] for edge in sg_edges]
    sg = nx.Graph()
    sg.add_edges_from(new_sg_edges)

    labels = [int(x in com) for x in allNodes]
    seed_labels = [int(x == seed) for x in allNodes]
    assert sum(seed_labels) == 1
    newcom = [rmapper[i] for i in range(len(labels)) if labels[i] == 1]
    miss = len(com) - sum(labels)
    # print(sorted(newcom),'\n',sorted(com))
    assert set(newcom).issubset(set(com))

    new_sg_edges_T = torch.tensor(new_sg_edges).T
    if all_edges:
        nodes = torch.tensor(list(sg.nodes()))
        src_nodes = torch.cartesian_prod(nodes, nodes)
        dgl_graph = dgl.graph(([], []), num_nodes=len(nodes))
        dgl_graph.add_edges(src_nodes[:, 0], src_nodes[:, 1])
    else:
        dgl_graph = dgl.add_self_loop(dgl.graph((new_sg_edges_T[0], new_sg_edges_T[1])))

    if pos_enc_dim is not None:
        h, ev = dgl.lap_pe(dgl_graph, pos_enc_dim, padding=True, return_eigval=True)
        # print(h,ev)
        dgl_graph.ndata['lap_pe'] = h
    if sample_pagerank is not None:
        probx = nx.pagerank(sg, personalization={mapper[seed]: 1}, alpha=alpha, max_iter=1000)
        pg = [item[1] for item in sorted(list(probx.items()))]
        dgl_graph.ndata['pg'] = torch.tensor(pg)
        if sample_pagerank == 'normal':
            dgl_graph.ndata['pg'] = (dgl_graph.ndata['pg'] - torch.mean(dgl_graph.ndata['pg'])) / torch.std(
                dgl_graph.ndata['pg'])
            # print(dgl_graph.ndata['pg'])
        if sample_pagerank == 'minmax':
            dgl_graph.ndata['pg'] = dgl_graph.ndata['pg'] / torch.max(dgl_graph.ndata['pg'])
            # print(dgl_graph.ndata['pg'])
    dgl_graph.ndata['x'] = torch.tensor([int(rmapper[int(x)] in com) for x in dgl_graph.nodes()])
    dgl_graph.ndata['seed'] = torch.tensor([int(rmapper[int(x)] == seed) for x in dgl_graph.nodes()])
    triangles = nx.triangles(sg)
    dgl_graph.ndata['triangles'] = torch.tensor([triangles[int(x)] for x in dgl_graph.nodes()])
    clustering_coefficients = nx.clustering(sg)
    dgl_graph.ndata['clustering_coefficients'] = torch.tensor(
        [clustering_coefficients[int(x)] for x in dgl_graph.nodes()]).float()
    if nodefeat is not None:
        node_index = [rmapper[int(x)] for x in dgl_graph.nodes()]
        dgl_graph.ndata['feat'] = torch.tensor(nodefeat[node_index])
    if shortest_path:
        dis = nx.shortest_path(sg, mapper[seed])
        dgl_graph.ndata['dis'] = torch.tensor([len(dis[int(x)]) - 1 for x in dgl_graph.nodes()])

    torch.save(
        {'dgl_graph': dgl_graph, 'rmapper': rmapper, 'labels': labels, 'seed': mapper[seed], 'com': com, 'miss': miss,
         'sg_size': len(sg_nodes), 'seed_labels': seed_labels, 'sg': sg},
        os.path.join('preprocess', f'{k}_{filename}'))
    print(k, "time consuming %0.2f" % (time() - st))


def mycollate(datas):
    batch_graph = []
    for i, data in enumerate(datas):
        data['dgl_graph'].ndata['index'] = torch.tensor([i] * len(data['dgl_graph'].nodes()))
        batch_graph.append(data['dgl_graph'])
    batch_graph = dgl.batch(batch_graph)
    keylist = list(datas[0].keys())
    keylist.remove('dgl_graph')
    batch_datas = {}
    for key in keylist:
        batch_datas[key] = [data[key] for data in datas]
    batch_datas['graph'] = batch_graph
    batch_datas['sg_size'] = torch.cumsum(torch.tensor([0] + batch_datas['sg_size']), dim=0)
    return batch_datas
