import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import random
import copy
import config
from autoencoder import GAE
from directed_train_test_split import direction_specific, direction_blind


edge_list = np.loadtxt('./newdataset/edgelist.txt', dtype=np.int64)
edge_index = torch.tensor(edge_list).t().contiguous()
num_nodes = len(set(edge_index.flatten().tolist()))
feature = pd.read_csv('./newdataset/1752morgan1024.csv', header=0, index_col=0)
features = torch.from_numpy(feature.values).to(torch.float32)


class DGAT_DDI(torch.nn.Module):
    def __init__(self, out_channels):
        super(DGAT_DDI, self).__init__()
        self.conv1 = GATConv(data.num_node_features, out_channels + config.beta_on, flow='source_to_target',
                             heads=16, concat=False, add_self_loops=False)
        self.conv2 = GATConv(data.num_node_features, out_channels + config.beta_on, flow='target_to_source',
                             heads=16, concat=False, add_self_loops=False)
        self.lin1 = torch.nn.Linear(data.num_node_features, 4 * out_channels)
        self.lin2 = torch.nn.Linear(4 * out_channels, out_channels)

    def forward(self, x, edge_index):
        F.dropout(x, p=0.6, training=self.training)
        x_in = F.elu(self.conv1(x, edge_index))
        x_out = F.elu(self.conv2(x, edge_index))
        x1 = F.elu(self.lin1(x))
        x_self = F.elu(self.lin2(x1))
        return x_in, x_out, x_self


def train():
    model.train()
    optimizer.zero_grad()
    z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z_in, z_out, z_self, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def testfinal(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def initialize_list():
    lists = [[] for _ in range(6)]
    return [lists[i] for i in range(6)]


target = ["auc", "ap", "acc", "f1", "pre", "re"]
auc_list, ap_list, f1_list, acc_list, pre_list, re_list = initialize_list()
target_list = [auc_list, ap_list,  f1_list, acc_list, pre_list, re_list]
for i in range(config.number):
    # config.seed = random.randint(0, 10000)

    # if i == 20:
    #     auc_list, ap_list, f1_list, acc_list, pre_list, re_list = initialize_list()
    #     target_list = [auc_list, ap_list, f1_list, acc_list, pre_list, re_list]
    #     config.task1 = False
    # if i > 20:
    #     config.task1 = False

    auc_l, ap_l, f1_l, acc_l, pre_l, re_l = initialize_list()
    target_l = [auc_l, ap_l, f1_l, acc_l, pre_l, re_l]
    for fold in range(config.fold):
        data = Data(edge_index=edge_index, num_nodes=num_nodes, x=features)
        if config.task1:
            data = direction_specific(data, fold, config.seed)
        else:
            data = direction_blind(data, fold, config.seed)
        print(data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        x = data.x.to(device)
        model = GAE(DGAT_DDI(config.out_channels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        min_loss_val = config.min_loss_val
        best_model = None
        min_epoch = config.min_epoch
        for epoch in range(1, config.epochs + 1):
            loss = train()
            if epoch % 10 == 0:
                auc, ap, acc, f1, pre, re = test(data.val_pos_edge_index, data.val_neg_edge_index)
                print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f},'
                      .format(epoch, auc, ap, acc, f1, pre, re))
            if epoch > min_epoch and loss <= min_loss_val:
                min_loss_val = loss
                best_model = copy.deepcopy(model)
        model = best_model
        auc, ap, acc, f1, pre, re = testfinal(data.test_pos_edge_index, data.test_neg_edge_index)
        print('final. AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f},'
              .format(auc, ap, acc, f1, pre, re))

        for j in range(6):
            target_l[j].append(eval(target[j]))
    for j in range(6):
        target_list[j].append(np.mean(target_l[j]))
for j in range(6):
    print(np.mean(target_list[j]), np.std(target_list[j]))

