import torch
import dgl
from memorys import *
from layers import *
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear, ReLU
import torch.nn.functional as F
from mixer import Mixer_per_node
from torch.distributions.normal import Normal

class BPMOE(torch.nn.Module):

    def __init__(self, args, mixer_configs, edge_predictor_configs,dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(BPMOE, self).__init__()

        self.num_experts_gnn = args.num_experts_gnn
        self.num_experts_mlp = args.num_experts_mlp

        self.mixer=nn.ModuleList([Mixer_per_node(mixer_configs, edge_predictor_configs) for _ in range(self.num_experts_mlp)])
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        if gnn_param['arch'] == 'transformer_attention':
            self.layers = nn.ModuleList([TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined) for _ in range(self.num_experts_gnn)])
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param['dim_out']) 
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
        #moe-topk 
        self.num_experts = self.num_experts_gnn+self.num_experts_mlp+1
        self.w_gate = nn.Parameter(torch.zeros(self.dim_node*6, self.num_experts), requires_grad=True) #100*5+edge_predictor_configs['dim_in_node']
        self.w_noise = nn.Parameter(torch.zeros(self.dim_node*6, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.k = args.topk
        self.noisy_gating = True
        self.register_buffer("mean", torch.tensor([0.1]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        self.deg_coef = nn.Parameter(torch.zeros(1, 100, 2))
        nn.init.xavier_normal_(self.deg_coef)
    def forward(self, mfgs, node_feats,inputs, has_temporal_neighbors,subgraph_node_feats,degree_node,neg_samples=1):
        
        self.memory_updater(mfgs[0])
        b=mfgs[0][0]
        memory_feats=b.srcdata['h'][:b.num_dst_nodes()]
        memory_feats = torch.nn.functional.relu(self.dropout(memory_feats))
        memory_feats=torch.unsqueeze(memory_feats, dim=1)

        node_feats_src = node_feats[b.srcdata['ID'].long()[:b.num_dst_nodes()]].float()
        node_degree= degree_node[b.srcdata['ID'].long()[:b.num_dst_nodes()]]
        log_deg = torch.log(node_degree + 1).unsqueeze(-1)
        recent_out=[] 
        spatial_out=[]

        for i in range(self.num_experts_mlp):
            recent_feats=self.mixer[i](inputs, has_temporal_neighbors, neg_samples, subgraph_node_feats)
            recent_feats = torch.nn.functional.relu(self.dropout(recent_feats))
            recent_feats=torch.unsqueeze(recent_feats, dim=1)
            recent_out.append(recent_feats)
        result_recent= torch.cat(recent_out,dim=1)

        for i in range(self.num_experts_gnn):
            rst = self.layers[i](mfgs[0][0])
            rst =  torch.stack([rst, rst * log_deg], dim=-1)
            rst = (rst * self.deg_coef).sum(dim=-1)

            rst=torch.unsqueeze(rst, dim=1)
            spatial_out.append(rst)
        result_spa= torch.cat(spatial_out,dim=1)
        combine_feats=torch.cat([memory_feats,result_spa,result_recent],dim=1)
        
        result_spa=torch.mean(result_spa,dim=1)
        result_recent=torch.mean(result_recent,dim=1)
        x1=memory_feats.squeeze(1)+result_spa+result_recent
        x2=torch.mul(torch.mul(memory_feats.squeeze(1), result_spa),result_recent)
        cat_feats=torch.cat([memory_feats.squeeze(1),result_spa,result_recent,x1,x2,node_feats_src],dim=1)
        gates, load = self.noisy_top_k_gating(cat_feats, self.training)

        out = (combine_feats * gates.unsqueeze(-1)).sum(dim=1)
        importance = gates.sum(0)
        loss_coef=0.4
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        pos,neg=self.edge_predictor (out, neg_samples=neg_samples)
   
        return pos,neg,loss      
     
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def get_emb(self, mfgs, node_feats,inputs, has_temporal_neighbors,subgraph_node_feats,degree_node,neg_samples=1):
        self.memory_updater(mfgs[0])
        b=mfgs[0][0]
        memory_feats=b.srcdata['h'][:b.num_dst_nodes()]
        memory_feats = torch.nn.functional.relu(self.dropout(memory_feats))
        memory_feats=torch.unsqueeze(memory_feats, dim=1)

        node_feats_src = node_feats[b.srcdata['ID'].long()[:b.num_dst_nodes()]].float()
        node_degree= degree_node[b.srcdata['ID'].long()[:b.num_dst_nodes()]]
        log_deg = torch.log(node_degree + 1).unsqueeze(-1)
        recent_out=[] 
        spatial_out=[]

        for i in range(self.num_experts_mlp):
            recent_feats=self.mixer[i](inputs, has_temporal_neighbors, neg_samples, subgraph_node_feats)
            recent_feats = torch.nn.functional.relu(self.dropout(recent_feats))
            recent_feats=torch.unsqueeze(recent_feats, dim=1)
            recent_out.append(recent_feats)
        result_recent= torch.cat(recent_out,dim=1)

        for i in range(self.num_experts_gnn):
            rst = self.layers[i](mfgs[0][0])
            rst =  torch.stack([rst, rst * log_deg], dim=-1)
            rst = (rst * self.deg_coef).sum(dim=-1)

            rst=torch.unsqueeze(rst, dim=1)
            spatial_out.append(rst)
        result_spa= torch.cat(spatial_out,dim=1)
        combine_feats=torch.cat([memory_feats,result_spa,result_recent],dim=1)
        
        result_spa=torch.mean(result_spa,dim=1)
        result_recent=torch.mean(result_recent,dim=1)
        x1=memory_feats.squeeze(1)+result_spa+result_recent
        x2=torch.mul(torch.mul(memory_feats.squeeze(1), result_spa),result_recent)
        cat_feats=torch.cat([memory_feats.squeeze(1),result_spa,result_recent,x1,x2,node_feats_src],dim=1)
        gates, load = self.noisy_top_k_gating(cat_feats, self.training)

        out = (combine_feats * gates.unsqueeze(-1)).sum(dim=1)
        return out 
class NodeClassificationModel(torch.nn.Module):
    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

