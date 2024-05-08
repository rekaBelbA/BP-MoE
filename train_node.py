import argparse
import os
import hashlib
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx


from sklearn.datasets import load_iris
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--batch_size', type=int, default=600)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model', type=str,required=True, help='name of stored model to load')
parser.add_argument('--posneg', default=False, action='store_true', help='for positive negative detection, whether to sample negative nodes')
parser.add_argument('--seed', type=int, default=29) 

parser.add_argument('--max_edges', type=int, default=30)
parser.add_argument('--structure_hops', type=int, default=1) 
parser.add_argument('--structure_time_gap', type=int, default=2000) 
parser.add_argument('--num_experts_gnn', type=int, default=2) 
parser.add_argument('--num_experts_mlp', type=int, default=2) 
parser.add_argument('--topk', type=int, default=3) 
parser.add_argument('--extra_neg_samples', type=int, default=1)
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')


args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.data == 'WIKI' or args.data == 'REDDIT':
    args.posneg = True

import torch
import time
import random
import dgl
import numpy as np
import pandas as pd
from modules import *
from sampler import *
from utils import *
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
role = ldf['ext_roll'].values
labels = ldf['label'].values.astype(np.int64)
emb_file_name = hashlib.md5(str(torch.load(args.model, map_location=torch.device('cpu'))).encode('utf-8')).hexdigest() + '.pt'

if not os.path.isdir('embs'):
    os.mkdir('embs')
if not os.path.isfile('embs/' + emb_file_name):
    print('Generating temporal embeddings..')
    node_feats, edge_feats,g,df = load_feat(args.data)
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    args.train_edge_end=train_edge_end 
    args.val_edge_end=val_edge_end  

    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    mixer_configs = {
                'per_graph_size'  : args.max_edges,  #50
                'time_channels'   : 100, 
                'input_channels'  : edge_feats.shape[1], 
                'hidden_channels' : 100, 
                'out_channels'    : 100,
                'num_layers'      : 1,
                'use_single_layer' : False
            }
    edge_predictor_configs = {
            'dim_in_time': 100,
            'dim_in_node': node_feats.shape[1],
        }

    model = BPMOE(args,mixer_configs, edge_predictor_configs,gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param).cuda()
    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    

    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()
    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                sample_param['strategy']=='recent', sample_param['prop_time'],
                                sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)
    
    model.load_state_dict(torch.load(args.model))

    processed_edge_id = 0
    
    subgraph = pre_compute_subgraphs(args, g, df, sample_param, train_param,sampler, neg_link_sampler,mode='none')
    
    ind=-1
    cur_inds = 0
    Gra=nx.Graph()
    for i in range(node_feats.shape[0]):
       Gra.add_node(i)

    emb = list()
    neg_samples = 1
    for _, rows in tqdm(df.groupby(ldf.index // train_param['batch_size'])):
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
        ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
        for i,j in zip(rows.src.values,rows.dst.values):
            Gra.add_edge(i,j)
        degree_node=torch.tensor(list(dict(Gra.degree()).values())).cuda()
        ind+=1
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = len(rows) * 2
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(model,mfgs, node_feats, edge_feats)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])

        subgraph_data_list = subgraph[ind]
        mini_batch_inds = get_random_inds(len(subgraph_data_list), args.extra_neg_samples, args.eval_neg_samples)
        subgraph_data = [subgraph_data_list[i] for i in mini_batch_inds]
        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()        
        num_of_df_links = len(subgraph_data_list) // ( args.extra_neg_samples+2)   
        subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)                
        cur_inds += num_of_df_links
        all_inds, has_temporal_neighbors = [], []
        all_edge_indptr = subgraph_data['all_edge_indptr']

        for i in range(len(all_edge_indptr)-1):
            num_edges = all_edge_indptr[i+1] - all_edge_indptr[i]
            all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
            has_temporal_neighbors.append(num_edges>0)
        inputs = [
            subgraph_edge_feats.cuda(), 
            subgraph_edts.cuda(), 
            len(has_temporal_neighbors), 
            torch.tensor(all_inds).long()
        ]
        has_temporal_neighbors = [True for _ in range(len(has_temporal_neighbors))] 
        with torch.no_grad():
            ret = model.get_emb(mfgs,node_feats,inputs, has_temporal_neighbors,subgraph_node_feats,degree_node)[:len(rows)]

        emb.append(ret)

    emb = torch.cat(emb, dim=0)
    torch.save(emb, 'embs/' + emb_file_name)
    print('Saved to embs/' + emb_file_name)
else:
    print('Loading temporal embeddings from embs/' + emb_file_name)
    emb = torch.load('embs/' + emb_file_name)

model = NodeClassificationModel(emb.shape[1], args.dim, labels.max() + 1).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
role = torch.from_numpy(role).type(torch.int32)

class NodeEmbMinibatch():

    def __init__(self, emb, role, label, batch_size):
        self.role = role
        self.label = label
        self.batch_size = batch_size
        self.train_emb = emb[role == 0]
        self.val_emb = emb[role == 1]
        self.test_emb = emb[role == 2]
        self.train_label = label[role == 0]
        self.val_label = label[role == 1]
        self.test_label = label[role == 2]
        self.mode = 0
        self.s_idx = 0

    def shuffle(self):
        perm = torch.randperm(self.train_emb.shape[0])
        self.train_emb = self.train_emb[perm]
        self.train_label = self.train_label[perm]

    def set_mode(self, mode):
        if mode == 'train':
            self.mode = 0
        elif mode == 'val':
            self.mode = 1
        elif mode == 'test':
            self.mode = 2
        self.s_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 0:
            emb = self.train_emb
            label = self.train_label
        elif self.mode == 1:
            emb = self.val_emb
            label = self.val_label
        else:
            emb = self.test_emb
            label = self.test_label
        if self.s_idx >= emb.shape[0]:
            raise StopIteration
        else:
            end = min(self.s_idx + self.batch_size, emb.shape[0])
            curr_emb = emb[self.s_idx:end]
            curr_label = label[self.s_idx:end]
            self.s_idx += self.batch_size
            return curr_emb.cuda(), curr_label.cuda()

if args.posneg:
    role = role[labels == 1]
    emb_neg = emb[labels == 0].cuda()
    emb = emb[labels == 1]
    labels = torch.ones(emb.shape[0], dtype=torch.int64).cuda()
    labels_neg = torch.zeros(emb_neg.shape[0], dtype=torch.int64).cuda()
    neg_node_sampler = NegLinkSampler(emb_neg.shape[0])

minibatch = NodeEmbMinibatch(emb, role, labels, args.batch_size)
if not os.path.isdir('models'):
    os.mkdir('models')
save_path = 'models/node_' + args.model.split('/')[-1]
best_e = 0
best_acc = 0
for e in range(args.epoch):
    minibatch.set_mode('train')
    minibatch.shuffle()
    model.train()
    for emb, label in minibatch:
        optimizer.zero_grad()
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        loss = loss_fn(pred, label.long())
        loss.backward()
        optimizer.step()
    minibatch.set_mode('val')
    model.eval()
    accs = list()
    with torch.no_grad():
        for emb, label in minibatch:
            if args.posneg:
                neg_idx = neg_node_sampler.sample(emb.shape[0])
                emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
            pred = model(emb)
            if args.posneg:
                acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
            else:
                acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
            accs.append(acc)
        acc = float(torch.tensor(accs).mean())
    print('Epoch: {}\tVal acc: {:.4f}'.format(e, acc))
    if acc > best_acc:
        best_e = e
        best_acc = acc
        torch.save(model.state_dict(), save_path)
print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(save_path))
minibatch.set_mode('test')
model.eval()
accs = list()
with torch.no_grad():
    for emb, label in minibatch:
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        if args.posneg:
            acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
        else:
            acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
        accs.append(acc)
    acc = float(torch.tensor(accs).mean())
print('Testing acc: {:.4f}'.format(acc))