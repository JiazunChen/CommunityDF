import argparse
from torch.utils.data import DataLoader
from dataset import Mydata, mycollate
from load_data import load_data
from tqdm import tqdm
from utils import PlaceHolder
import torch.nn.functional as F
import networkx as nx
import torch
from time import time
from getcomm import calu
import os
import datetime
from diffusion_model import DenoisingDiffusion, my_one_hot, PredictComsize, to_dense
from utils import eval,set_seed,initialize_from_checkpoint,save_model
import numpy as np
import json
from locator import Locator

def tensor_normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def fast_info_nce_loss_emd(pred, labels, T, graph_indices, k, m, tau=0.1,selected_graphs_ratio=0.2):
    n = pred.shape[0]
    pred = pred.view(n, -1)
    labels = labels.view(n, -1)
    T = T.view(n, -1)
    allindex = torch.where(T<1)[0]
    sample_index = allindex

    cos_sim_matrix = torch.mm(tensor_normalize(pred), tensor_normalize(labels).T)
    sample_graph_indices = graph_indices[sample_index]
    same_graph_mask = (graph_indices.unsqueeze(0) == sample_graph_indices.unsqueeze(1)).float()
    other_graph_mask = (graph_indices.unsqueeze(0) != sample_graph_indices.unsqueeze(1)).float()
    rand_same = torch.rand((len(sample_index), n), device=pred.device)
    rand_other = torch.rand((len(sample_index), n), device=pred.device)
    same_graph_sorted = torch.argsort(rand_same * same_graph_mask + (1 - same_graph_mask) * float('1e9'), dim=1)
    other_graph_sorted = torch.argsort(rand_other * other_graph_mask + (1 - other_graph_mask) * float('1e9'), dim=1)

    # Select top k and m indices for negatives
    same_graph_negatives = same_graph_sorted[:, :k]
    other_graph_negatives = other_graph_sorted[:, :m]
    negative_indices = torch.cat((same_graph_negatives, other_graph_negatives), dim=1)
    negative_pairs = cos_sim_matrix[sample_index.unsqueeze(1), negative_indices]
    positive_pairs = cos_sim_matrix[sample_index, sample_index].unsqueeze(1)

    positive_exp = torch.exp(positive_pairs / tau)
    negative_exp = torch.exp(negative_pairs / tau)
    denominator = positive_exp + negative_exp.sum(dim=1, keepdim=True)
    pair_loss = -torch.log(positive_exp / denominator).squeeze(1)
    weighted_loss = T[sample_index].view(-1) * pair_loss
    loss = weighted_loss.sum() / len(sample_index)
    return loss


def train_step(model, batch, device, Discriminative_model=None, pre_model=None, valid=False):
    batch['graph'].ndata['x'] = my_one_hot(batch['graph'].ndata['x'], 1).float()
    num_nodes_per_graph = batch['graph'].batch_num_nodes()
    graph_indices = torch.repeat_interleave(torch.arange(len(num_nodes_per_graph)), num_nodes_per_graph).to(device)
    batch['graph'] = batch['graph'].to(device)
    dense_data, node_mask = to_dense(batch, batch['graph'].ndata['x'])
    X = dense_data.X.to(device)
    node_mask = node_mask.to(device)
    if model.args.guided and model.args.competitor != 'cond':
        pg, _ = to_dense(batch, batch['graph'].ndata['pg'].reshape(-1, 1))
        pg = pg.X.cpu()
    else:
        pg = None
    with torch.no_grad():
        if Discriminative_model is not None:
            Discriminative_dense_data, Discriminative_node_mask = to_dense(batch, batch['graph'].ndata['pg'])
            Discriminative_X = Discriminative_dense_data.X.to(device)
            Discriminative_node_mask = Discriminative_node_mask.to(device)
            Discriminative_inputX = Discriminative_X[Discriminative_node_mask].float().view(-1,1)
            Discriminative_inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
            Discriminative_inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
            Discriminative_inputD = (Discriminative_inputD - args.mean_triangles) / args.std_triangles
            Discriminative_inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)

            Discriminative_predX = Discriminative_model(batch['graph'], Discriminative_inputX, Discriminative_inputS,
                                                        Discriminative_inputD, Discriminative_inputC)
            # batch['graph'].ndata['pg'] = Discriminative_predX.reshape(-1)
            pg, _ = to_dense(batch, torch.sigmoid(Discriminative_predX))
            batch['graph'].ndata['pg'] = pg.X[Discriminative_node_mask].reshape(-1)
            pg = pg.X.cpu()

        if pre_model is not None:
            old_X, node_mask, _ = pre_model.sample_batch(batch, Discriminative_model)
            pg = old_X.cpu()

    noisy_data = model.apply_noise(X, node_mask, pg)
    inputT = noisy_data['t'].repeat(1, noisy_data['X_t'].shape[1])[node_mask].float().view(-1, 1)
    inputX = noisy_data['X_t'][node_mask].float().view(-1, 1)
    inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
    inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
    inputD = (inputD - args.mean_triangles) / args.std_triangles
    inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)
    predX = model(batch['graph'], inputX, inputS, inputT, inputD, inputC)
    # print(predX.dtype,noisy_data['epsX'].dtype)
    loss = F.mse_loss(predX, noisy_data['epsX'][node_mask].to(device).float())
    if model.args.contrast_loss and not valid:
        noisy_data_sample = model.apply_noise(X, node_mask, pg)
        sample_inputT = noisy_data_sample['t'].repeat(1, noisy_data_sample['X_t'].shape[1])[node_mask].float().view(-1, 1)
        sample_inputX = noisy_data_sample['X_t'][node_mask].float().view(-1, 1)
        sample_emd = model.get_emd(batch['graph'], sample_inputX, inputS, sample_inputT, inputD, inputC)
        pred_emd = model.get_emd(batch['graph'], inputX, inputS, inputT, inputD, inputC)
        closs =  model.args.contrast_alpah*fast_info_nce_loss_emd(pred_emd, sample_emd, (1-torch.abs(sample_inputT-inputT)), graph_indices,
                                                         model.args.contrast_negative_sample_in_graph, model.args.contrast_negative_sample_out_graph,
                                                         model.args.contrast_tau,1).view(loss.shape)

        print('Contrast_loss',closs)
        loss += closs
    return loss


def normalize(X, norm_values, norm_biases, node_mask):
    X = (X - norm_biases) / norm_values
    return PlaceHolder(X).mask(node_mask)


def eval_bimatching_f1(method_name, comms, train_coms, test_coms, valid_size, mode='valid'):
    avglen = sum([len(com) for com in comms]) / len(comms)
    if mode == 'valid':
        bf, bj = calu(comms, test_coms[:valid_size])  # +train_coms)
    elif mode == 'test':
        bf, bj = calu(comms, test_coms + train_coms)
    f, j = calu(train_coms + test_coms, comms)  # +train_coms

    return f, bf, avglen, j, bj


def f1_score_(comm_find, comm):
    avglen = sum([len(com) for com in comm_find]) / len(comm_find)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for predicted, ground_truth in zip(comm_find, comm):
        true_positives = len(set(predicted) & set(ground_truth))
        p = true_positives / len(predicted) if predicted else 0
        r = true_positives / len(ground_truth) if ground_truth else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        total_precision += p
        total_recall += r
        total_f1 += f1
    avg_precision = total_precision / len(comm_find) if comm_find else 0
    avg_recall = total_recall / len(comm_find) if comm_find else 0
    avg_f1 = total_f1 / len(comm_find) if comm_find else 0
    return avg_f1, avg_precision, avg_recall, avglen


def train(args, graph, train_coms, test_coms, device, message="", features=None):
    set_seed(42)
    print(args)
    nowtime = datetime.datetime.now()
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists('./preprocess'):
        os.mkdir('./preprocess')
    args.save_path = './result/' + nowtime.strftime("%Y%m%d%H%M%S")
    if os.path.exists(args.save_path) == False:
        os.mkdir(args.save_path)

    st = time()
    traindata = Mydata(args.dataset, graph, args.sg_max_size, train_coms[args.locator_train_size:], 20, pos_enc_dim=args.pos_enc_dim,
                       shuffle=args.data_shuffle, \
                       sample_pagerank=args.sample_pagerank, cache=args.cache,
                       bfs_pagerank=args.bfs_pagerank, \
                       features=features, )
    triangles = []
    for data in traindata:
        triangles.extend(list(nx.triangles(data['sg']).values()))
    triangles = torch.tensor(triangles)
    args.mean_triangles = float(sum(triangles) / len(triangles))
    args.std_triangles = float((sum((x - args.mean_triangles) ** 2 for x in triangles) / len(triangles)) ** 0.5)
    gcntraindata = Mydata(args.dataset, graph, args.sg_max_size, train_coms[:args.locator_train_size]*args.locator_shuffle_time, 20, pos_enc_dim=args.pos_enc_dim,
                          shuffle=True,  \
                           sample_pagerank=args.sample_pagerank, cache=args.cache,
                           bfs_pagerank=args.bfs_pagerank, \
                           features=features, gcn_use=True)
    validdata = Mydata(args.dataset, graph, args.sg_max_size, test_coms[:args.valid_size], 20, mode='test',
                       pos_enc_dim=args.pos_enc_dim, \
                       sample_pagerank=args.sample_pagerank, cache=args.cache,
                       bfs_pagerank=args.bfs_pagerank, \
                       features=features)
    testdata = Mydata(args.dataset, graph, args.sg_max_size, test_coms[:], 20, mode='test',
                      pos_enc_dim=args.pos_enc_dim, \
                      sample_pagerank=args.sample_pagerank, cache=args.cache, bfs_pagerank=args.bfs_pagerank,
                      features=features)
    print('end time', time() - st)
    loader = {
        'train': DataLoader(traindata, collate_fn=mycollate, batch_size=args.batch_size, shuffle=args.loader_shuffle,
                            num_workers=0),
        'test': DataLoader(testdata, collate_fn=mycollate, batch_size=args.test_batch_size, num_workers=0),
        'valid': DataLoader(validdata, collate_fn=mycollate, batch_size=args.test_batch_size, num_workers=0),
    }
    model = DenoisingDiffusion(args, args.dim, args.dim, args.gatlayers, device=device, dropout=args.dropout,
                               diffusion_steps=args.diffusion_steps).to(device)

    if args.Discriminative_model_path!='' and args.guided:
        Discriminative_model = PredictComsize(args, args.dim, args.dim, args.dropout, device).to(device)
        Discriminative_model.load_state_dict(torch.load(args.Discriminative_model_path))
        Discriminative_model.eval()
    else:
        Discriminative_model = None

    if args.self_improve_path!='':
        checkpoint = torch.load(
            os.path.join(args.pre_model_path[0], f'{args.pre_model_path[2]}_checkpoint_{args.pre_model_path[1]}', ))
        pre_args = checkpoint['args']
        pre_model = DenoisingDiffusion(pre_args, pre_args.dim, pre_args.dim, pre_args.gatlayers, device=device,
                                       dropout=pre_args.dropout,
                                       diffusion_steps=pre_args.diffusion_steps).to(device)
        pre_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        pre_model.eval()
    else:
        pre_model = None

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    initmap = {}

    for th in range(1, 11):
        th = th / 10
        initmap[f'heap_com_threshold+{th}'] = 0
        if th != 1:
            th = int((th + 0.05) * 100) / 100
            initmap[f'heap_com_threshold+{th}'] = 0

    bestf, beste, bestlen, bestbf, bestff = initmap.copy(), initmap.copy(), initmap.copy(), initmap.copy(), initmap.copy()
    allbestf, allbeste, allbestmethod = 0, 100, 0

    if args.method_names == 'all':
        method_names = initmap.keys()
    else:
        assert args.method_names in initmap.keys()
        method_names = [args.method_names]
    best_eval_loss = 1000
    for e in tqdm(range(args.epoch)):
        allloss = 0
        model.train()
        for i, batch in enumerate(loader['train']):
            loss = train_step(model, batch, device, Discriminative_model, pre_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            allloss += loss.data
        print(f'epoch {e}  loss {allloss / (i + 1):.4f}')
        allloss = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader['valid']):
                # continue
                loss = train_step(model, batch, device, Discriminative_model, pre_model,valid=True)
                allloss += loss.data
            validloss = allloss / (i + 1)
        print(f'valid epoch {e}  validloss {validloss:.4f}  bestvalidloss {best_eval_loss:.4f}')
        if e >= args.eval_epoch_t and validloss <= best_eval_loss:
            best_eval_loss = validloss
            pred_comms = eval(args, model, loader['valid'], initmap, method_names, Discriminative_model, pre_model)
            print(pred_comms.keys())
            for method_name in method_names:
                method_com = pred_comms[method_name]
                f1, pre, rec, avglen = f1_score_(method_com, test_coms[:args.valid_size])
                if avglen <= 3:
                    continue
                print(
                    f'{method_name:25}  len {len(method_com):5d} pre:{pre:.4f} rec:{rec:.4f} f:{f1:.4f} bestf {bestf[method_name]:.4f} avglen:{avglen:.2f}')
                if f1 > bestf[method_name]:
                    save_model(model, args, args.save_path, e, method_name)
                    bestff[method_name] = pre
                    bestbf[method_name] = rec
                    bestf[method_name] = f1
                    beste[method_name] = e
                    bestlen[method_name] = avglen
                if f1 > allbestf:
                    allbestf = f1  # (f + bf) / 2
                    allbeste = e
                    allbestmethod = method_name

    print(f'message {message} args {args}')
    for method_name in [allbestmethod]:
        print(
            f'{method_name:25} beste {beste[method_name]:3d} bestff {bestff[method_name]:.4f} bestbf {bestbf[method_name]:.4f} bestf {bestf[method_name]:.4f} bestlen {bestlen[method_name]:.2f}')
    with open(args.log, 'a') as f:
        print(f'message {message} args {args}', file=f)
        print(f'valid allbestf {allbestf} allbeste {allbeste} method_name {allbestmethod}')
        print(f'valid allbestf {allbestf} allbeste {allbeste} method_name {allbestmethod}', file=f)
        for method_name in [allbestmethod]:
            print(
                f'{len(method_com):5} {method_name:25} beste {beste[method_name]:3d} bestff {bestff[method_name]:.4f} bestbf {bestbf[method_name]:.4f} bestf {bestf[method_name]:.4f} bestlen {bestlen[method_name]:.2f}\n',
                file=f)

    with open(args.log, 'a') as file:
        print(f'test allbestf {allbestf} allbeste {allbeste} method_name {allbestmethod}')
        print(f'test allbestf {allbestf} allbeste {allbeste} method_name {allbestmethod}', file=file)
        model, _ = initialize_from_checkpoint(os.path.join(args.save_path, f'{allbestmethod}_checkpoint_{allbeste}', ),
                                              model)
        best_method_t = float(allbestmethod.split('+')[1])
        method_names = [f'heap_com_threshold_nonc+{best_method_t}', f'heap_com_threshold+{best_method_t}']
        test_time = time()
        pred_comms = eval(args, model, loader['test'], initmap, method_names, Discriminative_model, pre_model)
        test_time = time() - test_time
        for method_name in method_names:
            method_com = pred_comms[method_name]
            f, bf, avglen, j, bj = eval_bimatching_f1(method_name, method_com, train_coms, test_coms, 0, 'test')
            if avglen <= 3:
                continue
            bif = (f + bf) / 2
            bij = (j + bj) / 2
            print(
                f'{len(method_com):5}  {method_name:25}  bestff {f:.4f} bestbf {bf:.4f} bestf {bif:.4f} bestj {bij:.4f} bestlen {avglen:.2f} test_time {test_time}\n')
            print(
                f'{len(method_com):5}  {method_name:25}  bestff {f:.4f} bestbf {bf:.4f} bestf {bif:.4f} bestj {bij:.4f}  bestlen {avglen:.2f} test_time {test_time}\n',
                file=file)
    ####Locator

    # gcntraindata = Mydata(args.dataset, graph, args.sg_max_size, train_coms[:args.locator_train_size]*args.locator_shuffle_time, 20, pos_enc_dim=args.pos_enc_dim,
    #                       shuffle=True,  \
    #                        sample_pagerank=args.sample_pagerank, cache=args.cache,
    #                        bfs_pagerank=args.bfs_pagerank, \
    #                        features=features, gcn_use=True)
    log= f"size {args.locator_train_size}*{args.locator_shuffle_time}"
    print('init gcn')
    locator = Locator(args,device,train_coms[:args.locator_train_size]*args.locator_shuffle_time,test_coms,gcntraindata,validdata,testdata,model,None,gcn_dropout=args.locator_dropout,gcn_hidden_size=args.locator_hiddensize,log=log)
    print('init gcn train')
    locator.train()
    print('init gcn eavl')
    locator.eval()

def initargs(parser):
    parser.add_argument('--dataset', type=str, default='facebook')
    parser.add_argument('--root', type=str, default='datasets')
    parser.add_argument('--train_size', type=int, default=50)
    parser.add_argument('--valid_size', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--sg_max_size', type=int, default=100)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=201)
    parser.add_argument('--diffusion_steps', type=int, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log', type=str, default='result.txt')
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--gatlayers', type=int, nargs='+', default=[4, 4, 4])
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--eval_epoch_t', type=int, default=5)
    parser.add_argument('--pos_enc_dim', type=int, default=200)
    parser.add_argument('--method_names', type=str, default='all')
    parser.add_argument('--prob_normal', type=str, default='minmax')
    parser.add_argument('--data_shuffle', action='store_true')
    parser.add_argument('--loader_shuffle', action='store_true')
    parser.add_argument('--bfs_pagerank', action='store_true', default=True)
    parser.add_argument('--sample_pagerank', type=str, default='normal')
    parser.add_argument('--cache', action='store_true', default=True)
    parser.add_argument('--attn_pool', action='store_true', default=True)
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--competitor', type=str, default=None)  # cond or  grad
    parser.add_argument('--self_improve_path', type=str, default='')
    parser.add_argument('--Discriminative_model_path', type=str, default='')
    parser.add_argument('--guided', action='store_true', default=True)
    parser.add_argument('--ori_xc_coefficient', action='store_true', default=False)
    parser.add_argument('--sample_place', type=str, default='all')  # None first nolabel all
    ####
    parser.add_argument('--contrast_loss', action='store_true', default=True) # None first nolabel all
    parser.add_argument('--contrast_negative_sample_out_graph', type=int, default=5)
    parser.add_argument('--contrast_negative_sample_in_graph', type=int, default=5)
    parser.add_argument('--contrast_alpah', type=float, default=0.1)
    parser.add_argument('--contrast_tau', type=float, default=1)
    ####
    parser.add_argument('--locator_train_size', type=int, default=50) #10 for Facebook
    parser.add_argument('--locator_shuffle_time', type=int, default=3)
    parser.add_argument('--locator_dropout', type=float, default=0)
    parser.add_argument('--locator_hiddensize', type=int, default=64)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = initargs(parser)
    args.dataset = 'new_' + args.dataset
    graph, train_coms, test_coms, features = load_data(args.root, args.dataset, args.train_size)
    if features is not None:
        args.feat_dim = features.shape[1]
    else:
        args.feat_dim = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.diffusion_steps ==-1:
        for args.diffusion_steps in range(5,31,5):
            train(args, graph, train_coms, test_coms, device, features=features)
    else:
        train(args, graph, train_coms, test_coms, device, features=features)
    print('end')
