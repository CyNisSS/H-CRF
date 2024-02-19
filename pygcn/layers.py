import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import pygcn.utils

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class U_CRF(Module):
    def __init__(self,in_dim,out_dim,layers,Nlist,**kwargs):
        super(U_CRF, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.NumList = Nlist
        self.crfs = nn.ModuleList()
        self.layers = layers # len(A list)
        for i in range(self.layers):
            self.crfs.append(CRF_NN(in_dim,out_dim,2,Nlist[i]))

    def forward(self,x,A_list,T_list,edge_list):
        x_list = []
        L = len(T_list) #4
        x_list.append(x)
        for i in range(len(T_list)):
            x = self.crfs[i](x,A_list[i],edge_list[i])
            # drop
            x = T_list[i].t().matmul(x)
            x_list.append(x)

        for i in range(L):#0,1,2,3
            x = T_list[L-i-1].matmul(x_list[L-i])+ x_list[L-i-1]
        return x

class CRF_NN(Module):
    def __init__(self, in_dim, out_dim, num_iters, N,M, heads=4, **kwargs):
        super(CRF_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iters = num_iters
        self.heads = heads
        #        self.adj = adj
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.src = nn.Linear(in_dim, out_dim, bias=False)
        self.trg = nn.Linear(in_dim, out_dim, bias=False)

        self.nodes_dim = 0

        self.src_nodes_dim = 0
        self.trg_nodes_dim = 1
        '''self.scores_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.scores_trg = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.gcn_feature_proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.crf_out = nn.Linear(heads * out_dim, out_dim, bias=False)'''
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = nn.ELU()
        self.add_skip_connection = True
        self.concat = True
        self.dropout = nn.Dropout(p=0.5)
        '''self.score_fn_source = nn.Parameter(torch.Tensor(1, out_dim))
        #self.score_fn_target = nn.Parameter(torch.Tensor(1, out_dim))'''
        self.edge_embedding = nn.Embedding(N, in_dim)
        self.node_embedding = nn.Embedding(M,in_dim)

        self.log_attention_weights = False  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        self.bias = nn.Parameter(torch.Tensor(heads * out_dim))

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.src.weight)
        nn.init.xavier_uniform_(self.trg.weight)
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    def edge_process(self, edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, edge)
        size = list(edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=edge.dtype, device=edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, edge)
        return neighborhood_sums
    def forward(self, x, adj,edge_index,node_wgt):
        output = x  # x-shape NxFIN
        num_of_nodes = x.size(0)
        outdim = self.out_dim

        edge_embed = self.edge_embedding(edge_index).mean(0)
        src_embed = self.edge_process(edge_embed, edge_index[0], num_of_nodes)
        trg_embed = self.edge_process(edge_embed, edge_index[1], num_of_nodes)
        edge_embed = F.softmax(torch.matmul(src_embed,trg_embed.t()),dim=1)

        node_embed = self.node_embedding(node_wgt)
        #x = torch.concat((x,node_embed),dim=1) # 2708x32
        #x = x+node_embed
        '''src_idx = edge_index[0]
        trg_idx = edge_index[1]
        src_embed = self.edge_embedding(src_idx)# 9104x16
        trg_embed = self.edge_embedding(trg_idx)
        print(trg_embed)
        src_embed_reduct = self.edge_process(src_embed,src_idx,num_of_nodes)# 3327 x 16
        trg_embed_reduct = self.edge_process(trg_embed,trg_idx,num_of_nodes)
        print('trg embed',trg_embed_reduct)
        edge_embed = F.normalize(torch.matmul(src_embed_reduct,trg_embed_reduct.t()), p=2, dim=0)
        #edge_embed = F.softmax(torch.matmul(src_embed_reduct, trg_embed_reduct.t()),dim=1)

        edge_features = src_embed + trg_embed# 9104 x 16
        edge_features = self.edge_process(edge_features,src_idx,num_of_nodes).sum(1)
        print('edge feature1 :',edge_features)
        edge_features = F.normalize(torch.matmul(edge_features,edge_features.t()),p=2,dim=0)
        #edge_features = F.normalize(torch.matmul(edge_features,edge_features.t()),p=2,dim=0)

        print('edge feature :',edge_features)

        edge = self.dropout((edge_embed+edge_features)*0.5)
        print('edge: ',edge)'''
        x1 = self.src(x)
        x2 = self.trg(x)
        '''gcn_features = self.gcn_feature_proj(x).view(-1,self.heads,self.out_dim)
        gcn_features = self.dropout(gcn_features)
        scores_src = (gcn_features * self.scores_src).sum(-1)
        print('scores_src shape:', scores_src.shape)
        scores_trg = (gcn_features * self.scores_trg).sum(-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_src, scores_trg, gcn_features, edge_index)
        print('after lifted--scores_src : nodes_features_proj:', scores_source_lifted,nodes_features_proj_lifted)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        gcn_features_proj_lifted = nodes_features_proj_lifted * attentions_per_edge

        out_nodes_features = self.aggregate_neighbors(gcn_features_proj_lifted,edge_index,x,num_of_nodes)
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, x, out_nodes_features)
        return out_nodes_features'''
        '''scores_source = (x * self.score_fn_source).sum(-1) # Nx FOUT ->Nx1
        scores_target = (x * self.score_fn_target).sum(-1)
        source_lifted,target_lifted,node_features_lifted = self.lift(scores_source,scores_target,x,edge_index)
        scores_per_edge = self.leakyReLU(source_lifted + target_lifted)
        att_per_edge = self.neighborhood_aware_softmax(scores_per_edge,edge_index[1],num_of_nodes)
        att_per_edge = self.dropout(att_per_edge)

        node_features_weighted = att_per_edge * node_features_lifted
        output = self.aggregate_neighbors(node_features_weighted,edge_index,x,num_of_nodes)
'''
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        logits = torch.matmul(x1, x2.t()) #* edge_embed
        logits = logits.mm(edge_embed)
        #print('x1 x x2:',torch.matmul(x1, x2.t()))
        logits = logits - logits.max()
        similarity = F.softmax(self.leakyReLU(logits), dim=1)
        #print('simi:', similarity)
        support = torch.mm(adj, similarity)  # NxN
        normalize = torch.sum(support, dim=1)
        #print('normalization : ', normalize)
        normalize = normalize.unsqueeze(-1).expand_as(x)  # Nxin_dim
        for i in range(self.num_iters):
            #output = self.norm((x * beta + (torch.mm(support, output) + output) * alpha) / (beta + normalize * alpha + alpha))
            output = (beta * x + (torch.mm(support, output)) * alpha) / (beta + normalize * alpha)
        #print('alpha: ',alpha)
        #print('beta: ',beta)
        return output
    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted
    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax
        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)
        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)
    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)
    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features
    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.heads, self.out_dim)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.heads * self.out_dim)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias
        print(out_nodes_features.shape)
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
