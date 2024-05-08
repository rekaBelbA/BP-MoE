import argparse
import os
import random
import time
import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import *
from sampler import *
from utils import *
import matplotlib.pyplot as plt
import networkx as nx

def get_low_training(df, train_edge_end, val_edge_end,data_num): 
    train_df = df[:train_edge_end]
    val_df = df[train_edge_end:val_edge_end]
    test_df = df[val_edge_end:]
    num_rows = len(train_df)
    num_rows_to_remove = int(num_rows * (1-data_num))
    random_indices = train_df.sample(num_rows_to_remove).index
    new_train_df = train_df.drop(random_indices)


    inductive_inds = []
    for index, (_, row) in enumerate(new_train_df.iterrows()):
            inductive_inds.append(index)
    new_train_edge_end=len(inductive_inds)

    for index, (_, row) in enumerate(val_df.iterrows()):
            inductive_inds.append(train_edge_end+index)
    new_val_edge_end=len(inductive_inds)

    for index, (_, row) in enumerate(test_df.iterrows()):
            inductive_inds.append(val_edge_end+index)    
    print('Inductive links', len(inductive_inds))
    
    return inductive_inds,new_train_edge_end,new_val_edge_end

def get_inductive_links(df, train_edge_end, val_edge_end): 
    train_df = df[:train_edge_end]
    val_df = df[train_edge_end:val_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    n_total_unique_nodes = len(total_node_set)

    test_node_set = set(np.unique(np.hstack([test_df['src'].values, test_df['dst'].values])))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))# 

    inductive_inds = []
    for index, (_, row) in enumerate(train_df.iterrows()):
        if row.src not in new_test_node_set and row.dst not in new_test_node_set:
            inductive_inds.append(index)
    new_train_edge_end=len(inductive_inds)

    for index, (_, row) in enumerate(val_df.iterrows()):
        if row.src in new_test_node_set or row.dst in new_test_node_set:
            inductive_inds.append(train_edge_end+index)
    new_val_edge_end=len(inductive_inds)

    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_test_node_set or row.dst in new_test_node_set:
            inductive_inds.append(val_edge_end+index)    
    print('Inductive links', len(inductive_inds))
    
    return inductive_inds,new_train_edge_end,new_val_edge_end

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eval(args,valid_subgraphs,test_subgraphs,Gra,mode='val'):

    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
        cur_inds = args.train_edge_end
        subgraph=valid_subgraphs
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
        cur_inds = args.val_edge_end
        subgraph=test_subgraphs
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        ind=-1
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
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
            
            pred_pos, pred_neg,loss = model(mfgs,node_feats,inputs, has_temporal_neighbors,subgraph_node_feats,degree_node)
            total_loss+=loss
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True,help='dataset name')
parser.add_argument('--config', type=str, required=True, help='config path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='BPMOE', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=1, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1)
parser.add_argument('--hidden_dropout_prob', type=float,default=0.1)
parser.add_argument('--num_hidden_layers', type=int,default=12)
parser.add_argument('--num_attention_heads', type=int,default=12)
parser.add_argument('--hidden_size', type=int,default=100)
parser.add_argument('--extra_neg_samples', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=600)
parser.add_argument('--max_edges', type=int, default=30)
parser.add_argument('--structure_hops', type=int, default=1) 
parser.add_argument('--structure_time_gap', type=int, default=2000) 
parser.add_argument('--seed', type=int, default=29) 
parser.add_argument('--num_experts_gnn', type=int, default=2) 
parser.add_argument('--num_experts_mlp', type=int, default=2) 
parser.add_argument('--topk', type=int, default=3) 
parser.add_argument('--data_num', type=float, default=1) 
parser.add_argument('--low_resource', default=False)

args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed) 

node_feats, edge_feats,g, df = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

if args.use_inductive:
    inductive_inds, train_edge_end,val_edge_end = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]
if args.low_resource:
   inductive_inds, train_edge_end,val_edge_end= get_low_training(df, train_edge_end, val_edge_end,args.data_num)
   df = df.iloc[inductive_inds] 
   
args.train_edge_end=train_edge_end 
args.val_edge_end=val_edge_end  

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

mixer_configs = {
            'per_graph_size'  : args.max_edges,  
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

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.dst.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    print(g['indptr'].shape[0])
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0]-1)
 
train_subgraphs = pre_compute_subgraphs(args, g, df, sample_param, train_param,sampler, neg_link_sampler,mode='train')
valid_subgraphs = pre_compute_subgraphs(args, g, df, sample_param, train_param,sampler, neg_link_sampler,  mode='valid')
test_subgraphs  = pre_compute_subgraphs(args, g, df, sample_param, train_param, sampler, neg_link_sampler, mode='test' )

if not os.path.isdir('models'):
    os.mkdir('models')
path_saver = 'models/{}_{}.pkl'.format(args.data, args.model_name)

best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
epoch_losses = [] 
set_seed(args.seed)
for e in range(train_param['epoch']):
    Gra=nx.Graph()
    for i in range(node_feats.shape[0]):
        Gra.add_node(i)
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    ind=-1
    cur_inds = 0
    for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
        ind+=1
        for i,j in zip(rows.src.values,rows.dst.values):
            Gra.add_edge(i,j)
        degree_node=torch.tensor(list(dict(Gra.degree()).values())).cuda()

        t_tot_s = time.time()
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = root_nodes.shape[0] * 2 // 3
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            time_sample += ret[0].sample_time()
        t_prep_s = time.time()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)

        mfgs = prepare_input(model,mfgs, node_feats, edge_feats)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        

        subgraph_data_list = train_subgraphs[ind]
        mini_batch_inds = get_random_inds(len(subgraph_data_list), args.extra_neg_samples, args.eval_neg_samples)
        subgraph_data = [subgraph_data_list[i] for i in mini_batch_inds]
        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]  #  b.edata['f'] 
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()  # b.edata['dt']
       
        num_subgraphs = len(mini_batch_inds)
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
        t_prep_s = time.time()
        optimizer.zero_grad()
        pred_pos, pred_neg,moe_loss = model(mfgs,node_feats,inputs, has_temporal_neighbors,subgraph_node_feats,degree_node)
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        loss += moe_loss
        total_loss += float(loss) * train_param['batch_size']
        loss.backward()
        optimizer.step()
        
        
        if mailbox is not None:
            eid = rows['Unnamed: 0'].values
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
        
        time_prep += time.time() - t_prep_s
        time_tot += time.time() - t_tot_s

    ap, auc = eval(args,valid_subgraphs,test_subgraphs,Gra,'val')
    if  ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))
    epoch_losses.append(total_loss)

print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
ap, auc = eval(args,valid_subgraphs,test_subgraphs,Gra,'test')
if args.eval_neg_samples > 1:
    print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
else:
    print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
    if args.use_inductive:
        setting= 'inductive'
    else:
        setting='transductive'
    file=open('output.txt', 'a')
    file.write('\ttest AP:{:4f}  test AUC:{:4f} spatial expert: {} behavior expert: {} Dataset: {}  \n'.format(ap, auc, args.num_experts_gnn,args.num_experts_mlp,args.data))
