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

from pygcn.coarsen import *
from pygcn.utils import * # load_data, accuracy,preprocess_features,origin_load_data,encode_onehot,preprocess_adj
from pygcn.models import GCN
from scipy.linalg import fractional_matrix_power
from datetime import datetime
import statistics
import enum
# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig

from sklearn.manifold import TSNE
from scipy.stats import entropy

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
parser.add_argument('--model_path', type=str, default='./models/')
parser.add_argument('--acc_dict', type=dict, default={'cora':0.828,'citeseer':0.721, 'pubmed':0.8})




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


'''print("\n")
print('layer_index ', 1)
print('input shape:   ', features.shape)
print('adj list\n ',adj_list)
print('transfer list\n  ',transfer_list)
'''
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

def test():
    model.eval()
    output,origin_x,crf_x = model(features, A_list,T_list,edge_list,node_wgt_list)
    output = F.log_softmax(output,dim=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return output,origin_x,crf_x

fig_tag = str(np.random.rand(1).item())[-4:-1]
chs_mod = 'citeseer--0.726'
mod_dir = '../models/'
new_m_pth = mod_dir+chs_mod+'.pt'
new_m_load = torch.load(new_m_pth)
model.load_state_dict(new_m_load)
output,origin_x,crf_x=test()
# Let's define an enum as a clean way to pick between different visualization options
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
class VisualizationType(enum.Enum):
    ATTENTION = 0,
    EMBEDDINGS = 1,
    ENTROPY = 2,
def visualize_gat_properties(model_name, dataset_name,output,edge_index,
                             visualization_type=VisualizationType.ATTENTION):
    # Fetch the data we'll need to create visualizations
    #all_nodes_unnormalized_scores, edge_index, node_labels, gat = gat_forward_pass(model_name, dataset_name)
    #edge_index = edge_list[0]
    output = output.detach().numpy()
    print('node_scores shape:', output.shape)
    print('edge list shape:', edge_index.shape)

    # Perform a specific visualization (t-SNE or entropy histograms)
    if visualization_type == VisualizationType.EMBEDDINGS:  # visualize embeddings (using t-SNE)
        node_labels = labels.cpu().numpy()
        print('node_labels shape:', node_labels.shape)
        num_classes = len(set(node_labels))

        t_sne_embeddings = TSNE(n_components=2,
                                perplexity=30, method='barnes_hut').fit_transform(output)

        fig = plt.figure(figsize=(12, 8), dpi=80)  # otherwise plots are really small in Jupyter Notebook
        for class_id in range(num_classes):
            # We extract the points whose true label equals class_id and we color them in the same way, hopefully
            # they'll be clustered together on the 2D chart - that would mean that GAT has learned good representations!
            plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1],
                        s=20, color=cora_label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
        #fig_name = 'gcn-'+chs_mod +'-'+fig_tag+'.png'
        fig_name = 'hidd-layer-crf-'+chs_mod +'-'+fig_tag+'.png'
        plt.savefig(fig_name)
        plt.show()
visualize_gat_properties(
        model,
        'citeseer',
        crf_x,
        edge_list[0],
        visualization_type=VisualizationType.EMBEDDINGS # pick between attention, t-SNE embeddings and entropy

)