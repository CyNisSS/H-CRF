from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import copy

from coarsen import *
from pygcn.utils import * # load_data, accuracy,preprocess_features,origin_load_data,encode_onehot,preprocess_adj
from pygcn.models import GCN
from scipy.linalg import fractional_matrix_power
from datetime import datetime
import statistics

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='citeseer')
parser.add_argument('--coarsen_level', type=int, default=2)
parser.add_argument('--max_node-wgt', type=int, default=50)
parser.add_argument('--exp_rounds', type=int, default=5)
parser.add_argument('--model_path', type=str, default='../models/')
parser.add_argument('--acc_dict', type=dict, default={'cora':0.828,'citeseer':0.725, 'pubmed':0.8})



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
"""adj, features, labels, idx_train, idx_val, idx_test = origin_load_data()"""

adj, features, y_train, y_val, y_test, idx_train, idx_val, idx_test,labels = load_data(args.dataset)

graph, mapping = read_graph_from_adj(adj, args.dataset)
print('total nodes:', graph.node_num)

# Step-1: Graph Coarsening.
original_graph = graph
transfer_list = []
adj_list = [copy.copy(graph.A)]
node_wgt_list = [copy.copy(graph.node_wgt)]
for i in range(args.coarsen_level):  # FLAGS.coarsen_level=4
    match, coarse_graph_size = generate_hybrid_matching(args.max_node_wgt, graph)  # FLAGS.max_node_wgt=50
    coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
    transfer_list.append(copy.copy(graph.C))
    graph = coarse_graph
    adj_list.append(copy.copy(graph.A))
    node_wgt_list.append(copy.copy(graph.node_wgt))
    print('There are %d nodes in the %d coarsened graph' % (graph.node_num, i + 1))


print("\n")
print('layer_index ', 1)
print('input shape:   ', features.shape)
print('adj list\n ',adj_list)
print('transfer list\n  ',transfer_list)

Nlist=[]
edge_list = []
for i in range(len(adj_list)):
    edge_list.append(get_edge_index(adj_list[i]))
    #adj_list[i] = [sparse_to_tuple(adj_list[i])] # +SNN using
    adj_list[i] = [preprocess_adj(adj_list[i])]

T_list = [] # csr_mx -> tensor
for i in range(len(transfer_list)):
    t = sp.coo_matrix(transfer_list[i])
    t = torch.FloatTensor(t.todense())
    T_list.append(t)
A_list = []
for i in range(len(adj_list)):
    a =  adj_list[i][0]
    a = tuples2tensor(a)
    # a = SNN(a)
    A_list.append(a)
    Nlist.append(a.shape[0])
    node_wgt_list[i] = torch.LongTensor(node_wgt_list[i])



nfeat = features.shape[1]
labels = torch.FloatTensor(labels)
labels=torch.argmax(labels,dim=1)
print(idx_train.shape)
print(idx_val.shape)
print(idx_test.shape)

'''adj = adj + np.eye(adj.shape[0])
row_sum = np.array(np.sum(adj,axis=1))
degree_matrix = np.matrix(np.diag(row_sum))
D = fractional_matrix_power(degree_matrix, -0.5)
adj = D.dot(adj).dot(D)
adj = torch.FloatTensor(adj)'''


# print(labels.max()+1)
# Model and optimizer
model = GCN(nfeat=nfeat,
            nhid=args.hidden,
            nclass=int(labels.max())+1 ,
            dropout=args.dropout,
            layers = args.coarsen_level,
            Nlist=Nlist,
            node_wgt_list=node_wgt_list)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    #adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    for i in range(len(T_list)):
        T_list[i] = T_list[i].cuda()
    for i in range(len(A_list)):
        A_list[i] = A_list[i].cuda()
        edge_list[i] = edge_list[i].cuda()
        node_wgt_list[i] = node_wgt_list[i].cuda()



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,origin_x,crf_x = model(features, A_list,T_list,edge_list,node_wgt_list)
    output = F.log_softmax(output, dim=1)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output,origin_x,crf_x = model(features, A_list, T_list, edge_list, node_wgt_list)
        output = F.log_softmax(output, dim=1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(i):
    model.eval()
    output,origin_x,crf_x = model(features, A_list,T_list,edge_list,node_wgt_list)
    output = F.log_softmax(output, dim=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
    line_str = 'data:%s ,coarsen_level: %d, epoch: %d,hidden_dim: %d,drop: %.2f ,experi_rounds: %d,test loss: %.5f,test acc: %.5f\n'
    with open('./result.txt', 'a+') as f:
        f.write(line_str % (
        args.dataset.upper(), args.coarsen_level, args.epochs, args.hidden, args.dropout, args.exp_rounds, loss_test, acc_test))

    return acc_test


# Train model
max_acc = 0
aver_acc = 0
cur_date = datetime.now().date()
acc_list = []
saved = set()
# Testing
for i in range(args.exp_rounds):
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    t = float(test(i+1))
    acc_list.append(t*1000)
    aver_acc = t + aver_acc
    max_acc = max(t,max_acc)
    model_pth = args.model_path + args.dataset
    if max_acc>=args.acc_dict[args.dataset] and max_acc not in saved:
        str_max = str(max_acc)
        model_pth = model_pth+'-GCN-'+str_max+'.pt'
        torch.save(model.state_dict(),model_pth)
        saved.add(max_acc)
aver_acc = aver_acc/args.exp_rounds
std = statistics.variance(acc_list)
line_str = 'MAX_ACC: %.4f, AVERAGE_ACC: %.4f(+-)%.6f, CURRENT_DATE: %s\n================================================================================================================\n'
with open('./result.txt', 'a+') as f:
    f.write(line_str % (max_acc,aver_acc,std/(1e6),cur_date))
