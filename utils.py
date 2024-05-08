import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torch_sparse import SparseTensor
import torch_sparse

#from modules.GeneralModel import *
def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    g, df = load_graph(d)
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        node_feats = node_feats.type(torch.float32)
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    else:
        node_feats=torch.randn(g['indptr'].shape[0] -1 ,100)
        node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
        elif d == 'Social':
            edge_feats = torch.randn(53931, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
    edge_feats = edge_feats.type(torch.float32)

    return node_feats, edge_feats, g, df

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
  
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
      
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
      
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
       
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def prepare_input(model,mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

def pre_compute_subgraphs(args, g, df, sample_param, train_param,sampler, neg_link_sampler,mode):
    ###################################################
    # get cached file_name
    if mode == 'train':
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1
    if args.use_inductive:
        setting="inductive"
    else:
        setting="transductive"
        
    fn = os.path.join(os.getcwd(), 'DATA', args.data, 
                        '%s_%s_neg_sample_neg%d_bs%d_hops%d_neighbors%d_seed_%d.pickle'%(mode,
                                                                                 setting, 
                                                                            extra_neg_samples, 
                                                                            train_param['batch_size'], 
                                                                            sample_param['layer'], 
                                                                            sample_param['neighbor'][0],
                                                                            args.seed))
    ###################################################

    # try:
    if os.path.exists(fn):
        all_subgraphs = pickle.load(open(fn, 'rb'))
        print('load ', fn)

    else:
        ###################################################
        # for each node, sample its neighbors with the most recent neighbors (sorted) 
        print('Sample subgraphs ... for %s mode'%mode)
        #sampler, neg_link_sampler = get_parallel_sampler(g, sample_param['neighbor'])
    
        ###################################################
        # setup modes
        if mode == 'train':
            cur_df = df[:args.train_edge_end]

        elif mode == 'valid':
            cur_df = df[args.train_edge_end:args.val_edge_end]

        elif mode == 'test':
            cur_df = df[args.val_edge_end:]
        else:
            cur_df =df
        loader = cur_df.groupby(cur_df.index // train_param['batch_size'])
        pbar = tqdm(total=len(loader))
        pbar.set_description('Pre-sampling: %s mode with negative sampleds %s ...'%(mode, extra_neg_samples))

        ###################################################
        all_subgraphs = []
        sampler.reset()
        for _, rows in loader:

            root_nodes = np.concatenate(
               [rows.src.values, 
                rows.dst.values, 
                neg_link_sampler.sample(len(rows) * extra_neg_samples)]
            ).astype(np.int32)

            # time-stamp for node = edge time-stamp
            ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
            #add for our model classification
            #root_nodes=rows.node.values.astype(np.int32)
            #ts=rows.time.values.astype(np.float32)
            all_subgraphs.append(get_mini_batch(sampler, root_nodes, ts, sample_param['layer']))
            
            pbar.update(1)
        pbar.close()

        try:
            pickle.dump(all_subgraphs, open(fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print('For some shit reason pickle cannot save ... but anyway ...')
        
        ###################################################
        
    print(len(all_subgraphs))
    return all_subgraphs

def get_mini_batch(sampler, root_nodes, ts, num_hops): # neg_samples is not used
    """
    Call function fetch_subgraph()
    Return: Subgraph of each node. 
    """
    all_graphs = []
    
    for root_node, root_time in zip(root_nodes, ts):
        all_graphs.append(fetch_subgraph(sampler, root_node, root_time, num_hops))

    return all_graphs

def fetch_subgraph(sampler, root_node, root_time, num_hops):
    """
    Sample a subgraph for each node or node pair
    """
    all_row_col_times_nodes_eid = []

    # suppose sampling for both a single node and a node pair (two side of a link)
    if isinstance(root_node, list):
        nodes, ts = [i for i in root_node], [root_time for i in root_node]
    else:
        nodes, ts = [root_node], [root_time]
    
    # fetch all nodes+edges
    for _ in range(num_hops):
        sampler.sample(nodes, ts)
        ret = sampler.get_ret() # 1-hop recent neighbors
        row, col, eid = ret[0].row(), ret[0].col(), ret[0].eid()
        nodes, ts = ret[0].nodes(), ret[0].ts().astype(np.float32)
        row_col_times_nodes_eid = np.stack([ts[row], nodes[row], ts[col], nodes[col], eid]).T
        all_row_col_times_nodes_eid.append(row_col_times_nodes_eid)
    all_row_col_times_nodes_eid = np.concatenate(all_row_col_times_nodes_eid, axis=0)

    # remove duplicate edges and sort according to the root node time (descending)
    all_row_col_times_nodes_eid = np.unique(all_row_col_times_nodes_eid, axis=0)[::-1]
    all_row_col_times_nodes = all_row_col_times_nodes_eid[:, :-1]
    eid = all_row_col_times_nodes_eid[:, -1]

    # remove duplicate (node+time) and sorted by time decending order
    all_row_col_times_nodes = np.array_split(all_row_col_times_nodes, 2, axis=1)
    times_nodes = np.concatenate(all_row_col_times_nodes, axis=0)
    times_nodes = np.unique(times_nodes, axis=0)[::-1]
    
    # each (node, time) pair identifies a node
    node_2_ind = dict()
    for ind, (time, node) in enumerate(times_nodes):
        node_2_ind[(time, node)] = ind

    # translate the nodes into new index
    row = np.zeros(len(eid), dtype=np.int32)
    col = np.zeros(len(eid), dtype=np.int32)
    for i, ((t1, n1), (t2, n2)) in enumerate(zip(*all_row_col_times_nodes)):
        row[i] = node_2_ind[(t1, n1)]
        col[i] = node_2_ind[(t2, n2)]
        
    # fetch get time + node information
    eid = eid.astype(np.int32)
    ts = times_nodes[:,0].astype(np.float32)
    nodes = times_nodes[:,1].astype(np.int32)
    dts = root_time - ts # make sure the root node time is 0
    
    return {
        # edge info: sorted with descending row (src) node temporal order
        'row': row, 
        'col': col, 
        'eid': eid, 
        # node info
        'nodes': nodes , # sorted by the ascending order of node's dts (root_node's dts = 0)
        'dts': dts,
        # graph info
        'num_nodes': len(nodes),
        'num_edges': len(eid),
        # root info
        'root_node': root_node,
        'root_time': root_time,
    }

def get_random_inds(num_subgraph, cached_neg_samples, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2+cached_neg_samples)
    #batch_size = num_subgraph 

    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
    neg_dst_inds = np.random.randint(low=2, high=2+cached_neg_samples, size=batch_size*neg_samples)
    neg_dst_inds = batch_size * neg_dst_inds + np.arange(batch_size)
    mini_batch_inds = np.concatenate([pos_src_inds, pos_dst_inds, neg_dst_inds]).astype(np.int32)
    ###################################################
    # for our model add below:
    #mini_batch_inds = np.arange(num_subgraph)
    return mini_batch_inds

def construct_mini_batch_giant_graph(all_graphs, max_num_edges):
    """
    Take the subgraph computed by fetch_subgraph() and combine it into a giant graph
    Return: the new indices of the graph
    """
    
    all_rows, all_cols, all_eids, all_nodes, all_dts = [], [], [], [], []
    
    cumsum_edges = 0
    all_edge_indptr = [0]
    
    cumsum_nodes = 0
    all_node_indptr = [0]
    
    all_root_nodes = []
    all_root_times = []
    for all_graph in all_graphs:
        # record inds
        num_nodes = all_graph['num_nodes']
        num_edges = min(all_graph['num_edges'], max_num_edges)
        
        # add graph information
        all_rows.append(all_graph['row'][:num_edges] + cumsum_nodes)
        all_cols.append(all_graph['col'][:num_edges] + cumsum_nodes)
        all_eids.append(all_graph['eid'][:num_edges])
        
        all_nodes.append(all_graph['nodes'])
        all_dts.append(all_graph['dts'])

        # update cumsum
        cumsum_nodes += num_nodes
        all_node_indptr.append(cumsum_nodes)
        
        cumsum_edges += num_edges
        all_edge_indptr.append(cumsum_edges)
        
        # add root nodes
        all_root_nodes.append(all_graph['root_node'])
        all_root_times.append(all_graph['root_time'])
    # for each edges
    all_rows = np.concatenate(all_rows).astype(np.int32)
    all_cols = np.concatenate(all_cols).astype(np.int32)
    all_eids = np.concatenate(all_eids).astype(np.int32)
    all_edge_indptr = np.array(all_edge_indptr).astype(np.int32)
    
    # for each nodes
    all_nodes = np.concatenate(all_nodes).astype(np.int32)
    all_dts = np.concatenate(all_dts).astype(np.float32)
    all_node_indptr = np.array(all_node_indptr).astype(np.int32)
        
    return {
        # for edges
        'row': all_rows, 
        'col': all_cols, 
        'eid': all_eids, 
        'edts': all_dts[all_cols] - all_dts[all_rows],
        # number of subgraphs + 1
        'all_node_indptr': all_node_indptr,
        'all_edge_indptr': all_edge_indptr,
        # for nodes
        'nodes': all_nodes, 
        'dts': all_dts, 
        # general information
        'all_num_nodes': cumsum_nodes,
        'all_num_edges': cumsum_edges,
        # root nodes
        'root_nodes': np.array(all_root_nodes, dtype=np.int32), 
        'root_times': np.array(all_root_times, dtype=np.float32), 
    }

def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    
    num_duplicate = len(root_nodes) // num_links 
    num_nodes = node_feats.shape[0]

    root_inds = torch.arange(len(root_nodes)).view(num_duplicate, -1)
    root_inds = [arr.flatten() for arr in root_inds.chunk(1, dim=1)]

    output_feats = torch.zeros((len(root_nodes), node_feats.size(1))).cuda()
    i = start_i
    ## add for our model below:
    #root_inds= torch.arange(len(root_nodes))
    #root_inds = [root_inds]

    for _root_ind in root_inds:

        if i == 0 or args.structure_hops == 0:
            sign_feats = node_feats.clone()
        else:
            prev_i = max(0, i - args.structure_time_gap)
            cur_df = df[prev_i: i] # get adj's row, col indices (as undirected)
            src = torch.from_numpy(cur_df.src.values)
            dst = torch.from_numpy(cur_df.dst.values)
            edge_index = torch.stack([
                torch.cat([src, dst]), 
                torch.cat([dst, src])
            ])

            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True) 

            mask = edge_index[0]!=edge_index[1] # ignore self-loops

            adj = SparseTensor(
                # value = edge_cnt[mask].float(), # take number of edges into consideration
                value = torch.ones_like(edge_cnt[mask]).float(),
                row = edge_index[0][mask].long(),
                col = edge_index[1][mask].long(),
                sparse_sizes=(num_nodes, num_nodes)
            )
            adj_norm = row_norm(adj).cuda()

            sign_feats = [node_feats]
            for _ in range(args.structure_hops):
                sign_feats.append(adj_norm@sign_feats[-1])
            sign_feats = torch.sum(torch.stack(sign_feats), dim=0)

        output_feats[_root_ind] = sign_feats[root_nodes[_root_ind]]

        i += len(_root_ind) // num_duplicate

    return output_feats
def row_norm(adj_t):
    if isinstance(adj_t, torch_sparse.SparseTensor):
        # adj_t = torch_sparse.fill_diag(adj, 1)
        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        return adj_t