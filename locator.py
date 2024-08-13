
from torch.utils.data import  DataLoader
from dataset import  mycollate
from tqdm import tqdm
import dgl
from utils import f1_score_,eval_bimatching_f1,set_seed
from gat import GCN

import torch

import torch.nn.functional as F


from getcomm import heap_com_threshold,heap_com_threshold_nonc,heap_com_topk,heap_com

def get_com(args,seeds,sg,probx,batch,method_name,predsize):
    comms = []
    com_method, threshold = method_name.split('+')
    threshold = float(threshold)
    # print(com_method,threshold)
    if com_method == 'heap_com_threshold':
        sg_coms = [heap_com_threshold(int(seed), sg, probx, size)for seed,size in zip(seeds,predsize)]
    elif com_method == 'heap_topk':
        sg_coms = [heap_com_topk(int(seed), sg, probx, size) for seed,size in zip(seeds,predsize)]
    elif com_method == 'heap':
        sg_coms = [heap_com(int(seed), sg, probx) for seed in seeds]
    elif com_method=='heap_com_threshold_nonc':
        sg_coms = [heap_com_threshold_nonc(int(seed), sg, probx, threshold,list(batch['rmapper'][index].keys()),int(batch['sg_size'][index])) for index,seed in enumerate(seeds) ]
    assert len(sg_coms) == len(batch['rmapper'])
    coms = []
    for index in range(len(sg_coms)):
        com = [batch['rmapper'][index][int(node - batch['sg_size'][index])] for node in sg_coms[index]]
        coms.append(com)
    for com in coms:
        if len(com):
            comms.append(com)
    return comms


class Locator():
    def __init__(self,args,device,train_coms,test_coms,traindata,validdata,testdata,pre_model,ics_model,post_batch_size=30,post_test_batch_size=100,gcnlr=1e-2,gcn_hidden_size=16,gcn_dropout=0.0,log=""):
        set_seed(42)
        self.args = args
        args.post_batch_size = post_batch_size
        args.post_test_batch_size = post_test_batch_size
        self.pre_model = pre_model
        self.pre_model.eval()
        self.ics_model = ics_model
        if self.ics_model is not None:
            self.ics_model.eval()
        self.loader = {}
        self.train_coms = train_coms
        self.test_coms = test_coms
        self.gcn_dropout =gcn_dropout
        self.gcn_hidden_size = gcn_hidden_size
        self.log =log
        self.traindata = self.update(traindata)
        self.loader['train'] = DataLoader(traindata, collate_fn=mycollate, batch_size=min(len(self.traindata),args.post_batch_size), shuffle=True,
                                     num_workers=0, drop_last=False)

        self.validdata = self.update(validdata)
        self.loader['valid'] = DataLoader(validdata, collate_fn=mycollate, batch_size=args.post_test_batch_size, shuffle=False,
                                     num_workers=0, drop_last=False)

        self.testdata = self.update(testdata)
        self.loader['test'] = DataLoader(testdata, collate_fn=mycollate, batch_size=args.post_test_batch_size, shuffle=False,
                                    num_workers=0)
        self.args.gcnlr =gcnlr
        self.gcnmodel  = GCN(2,gcn_hidden_size,1,gcn_dropout).to(device)
        self.device = device

    def valid(self):
        total_loss = 0

        self.gcnmodel.eval()
        pred_comms = []
        with torch.no_grad():
            for batch in (self.loader['valid']):
                batch['graph'] = batch['graph'].to(self.device)
                predX = batch['graph'].ndata['probx']
                predX = torch.hstack([predX,batch['graph'].ndata['seed'].view(-1,1)])
                predC = self.gcnmodel(batch['graph'], predX,batch['graph'].ndata['seed']==1)
                presize = predC.cpu()
                #print('presize',presize)
                sg = batch['graph'].cpu().to_networkx()
                seeds = torch.tensor(batch['seed']) + batch['sg_size'][:-1]
                pred_comms.extend(get_com(self.args, seeds, sg, batch['graph'].ndata['probx'].cpu(), batch, 'heap_com_threshold+0', presize))
        f1, pre, rec, avglen = f1_score_(pred_comms, self.test_coms[:len(pred_comms)])
        print('f1', f1, 'avglen', avglen)
        return f1

    def train(self,epoch = 100):
        optimizer = torch.optim.AdamW(self.gcnmodel.parameters(), lr=self.args.gcnlr)
        best_loss = 0
        best_e = 1000
        valid_loss = 100
        for epoch in tqdm(range(epoch)):
            total_loss = 0
            self.gcnmodel.train()
            for batch in (self.loader['train']):
                optimizer.zero_grad()
                batch['graph'] = batch['graph'].to(self.device)
                predX = batch['graph'].ndata['probx']
                predX = torch.hstack([predX,batch['graph'].ndata['seed'].view(-1,1)])
                predC = self.gcnmodel(batch['graph'], predX,batch['graph'].ndata['seed']==1)
                label = torch.tensor(batch['best_value']).view(-1, 1).to(self.device).float()
                loss = F.mse_loss(predC, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(total_loss / len(self.loader['train']))
            if total_loss / len(self.loader['train']) < 1:
                valid_loss = self.valid()
                print(epoch, total_loss / len(self.loader['train']), valid_loss, best_loss,best_e)
                if valid_loss > best_loss:
                    best_loss = valid_loss
                    best_e = epoch
                    torch.save(self.gcnmodel.state_dict(), 'gcn_model_test.pth')
            if epoch -best_e >20:
                break

    def eval(self):
        self.gcnmodel.load_state_dict(torch.load(f'gcn_model_test.pth'))
        total_loss = 0
        self.gcnmodel.eval()
        pred_comms = []
        with torch.no_grad():
            for batch in tqdm(self.loader['test']):
                batch['graph'] = batch['graph'].to(self.device)
                predX = batch['graph'].ndata['probx']
                predX = torch.hstack([predX,batch['graph'].ndata['seed'].view(-1,1)])
                predC = self.gcnmodel(batch['graph'], predX,batch['graph'].ndata['seed']==1)
                presize = predC.cpu()
                sg = batch['graph'].cpu().to_networkx()
                seeds = torch.tensor(batch['seed']) + batch['sg_size'][:-1]
                pred_comms.extend(get_com(self.args, seeds, sg, batch['graph'].ndata['probx'].cpu(), batch, 'heap_com_threshold+0', presize))
        method_name = self.log+f'gcn {self.gcn_dropout} {self.gcn_hidden_size}'
        f, bf, avglen, j, bj = eval_bimatching_f1(method_name, pred_comms, self.train_coms, self.test_coms, 0, 'test')
        with open(self.args.log, 'a') as file:
            bif = (f + bf) / 2
            bij = (j + bj) / 2
            print(
                f'{len(pred_comms):5}  {method_name:25}  bestff {f:.4f} bestbf {bf:.4f} bestf {bif:.4f} bestj {bij:.4f} bestlen {avglen:.2f}\n')
            print(
                f'{len(pred_comms):5}  {method_name:25}  bestff {f:.4f} bestbf {bf:.4f} bestf {bif:.4f} bestj {bij:.4f}  bestlen {avglen:.2f}\n',
                file=file)


    def update(self,dataset):
        ics_train_loader = DataLoader(dataset, collate_fn=mycollate, batch_size=self.args.post_batch_size, shuffle=False,
                                      num_workers=0)
        idx = 0
        for batch in ics_train_loader:
            probx = self.get_df_prob(batch)
            batch['graph'].ndata['probx'] = probx
            bg = dgl.unbatch(batch['graph'])
            for g in bg:
                # print(g)
                dataset[idx]['dgl_graph'].ndata['probx'] = g.ndata['probx'].cpu()
                idx += 1
        for idx, tmpdata in tqdm(enumerate(dataset)):
            seed = tmpdata['seed']
            sg = tmpdata['sg']
            probx = tmpdata['dgl_graph'].ndata['probx']
            size = sum(tmpdata['labels'])
            graph_size = len(tmpdata['labels'])
            # sg_coms = [heap_com_threshold(int(seed), sg, probx, threshold) for seed in seeds]
            best_k, best_f1, best_value = 0, 0, 0
            best_new_labels = []
            for size_bound in range(-10, 10):
                if size + size_bound >= 3:
                    sg_com = heap_com_topk(int(seed), sg, probx, size + size_bound)
                    new_labels = [1 if i in sg_com else 0 for i in range(graph_size)]
                    new_sg_com = [tmpdata['rmapper'][n] for n in sg_com]
                    f1, _, _, k = f1_score_([new_sg_com], [tmpdata['com']])
                    minvalue = min(probx[sg_com])
                    # print(f1,k,minvalue)
                    if best_f1 < f1:
                        best_f1 = f1
                        best_k = k
                        best_value = minvalue
                        best_new_labels = new_labels
            dataset[idx]['best_k'] = int(best_k)
            dataset[idx]['best_value'] = best_value
            dataset[idx]['best_new_labels'] = best_new_labels
        return dataset

    def get_df_prob(self,batch):
        self.pre_model.eval()
        with torch.no_grad():
            X, node_mask, _ = self.pre_model.sample_batch(batch, self.ics_model)
            #print(ics_inputX.shape)
            if self.args.prob_normal == 'softmax':
                X = X[node_mask]
                X = torch.sigmoid(X)
            elif self.args.prob_normal == 'minmax':
                for i in range(len(X)):
                    X[i] = (X[i] - min(X[i])) / (max(X[i]) - min(X[i]))
                X = X[node_mask]
        return X


