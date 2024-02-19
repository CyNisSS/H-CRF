import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution,CRF_NN,U_CRF



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,layers,Nlist,node_wgt_list):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.layers = layers # 4
        self.Numlist= Nlist
        #self.crf_nn = CRF_NN(nhid,nhid,2,10,10)
        #self.gcn = GraphConvolution(nhid,nhid)
        self.gcns = nn.ModuleList()
        self.crfs = nn.ModuleList()
        self.crfs_refine = nn.ModuleList()

        self.max_node_list = [i.max()+1 for i in node_wgt_list]
        for i in range(layers):
            self.crfs.append(CRF_NN(nhid,nhid,1,Nlist[i+1],self.max_node_list[i+1]))
            #self.crfs_refine.append(CRF_NN(nhid,nhid,2,Nlist[i+1],self.max_node_list[i+1]))

            self.gcns.append(GraphConvolution(nhid,nhid))
        #self.ucrf = U_CRF(nhid,nhid,layers,Nlist)

        '''self.Node_embeds = nn.ModuleList()
        for i in range(len(Nlist)):
            self.Node_embeds.append(nn.Embedding(max_node_list[i],nhid))'''
    def forward(self, x, A_list,T_list,edge_list,node_wgt_list):
        A0 = A_list[0]
        x_list = []
        x = F.relu(self.gc1(x, A0))
        gcn_hidden = x
        x_list.append(x)
        refine_list=[]
        for i in range(self.layers): # layers 4
            x = T_list[i].t().matmul(x)
            #x = self.gcns[i](x,A_list[i+1])
            #x = F.dropout(x, self.dropout, training=self.training)
            x = self.crfs[i](x,A_list[i+1],edge_list[i+1],node_wgt_list[i+1])
            x_list.append(x)
        # x 288x16, x_list 5 [2708,1447,830,482,288]
        l = self.layers-1
        refine_list.append(x_list[-1])
        x= x_list[-1]
        for i in range(self.layers):
            x = T_list[l - i].matmul(x) + x_list[l - i]
        crf_hidden = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A0)
        return x,gcn_hidden,crf_hidden
    '''def forward(self, x, A_list,T_list,edge_list):

        A0 = A_list[0]
        x = F.relu(self.gc1(x, A0))
        x = self.ucrf(x,A_list,T_list,edge_list)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A0)
        return F.log_softmax(x, dim=1)'''
