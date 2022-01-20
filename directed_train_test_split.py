import math
import torch
from torch_geometric.data import Data
import numpy as np
import config


def direction_specific(data, fold, seed, val_ratio=config.val_ratio, test_ratio=config.test_ratio):

    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    row_original, col_original = data.edge_index
    data.edge_index = None
    n_v = int(math.floor(val_ratio * row_original.size(0)))
    n_t = int(math.floor(test_ratio * row_original.size(0)))
    n_a = int(math.floor(row_original.size(0)))

    # Positive edges.
    perm = torch.randperm(row_original.size(0))
    start_step = int(fold / config.fold * perm.size().numel())
    perm_repeat = torch.cat([perm, perm], dim=0)
    row, col = row_original[perm_repeat], col_original[perm_repeat]
    r, c = row[start_step:start_step + n_v], col[start_step:start_step + n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)    # 将前20%划分为验证集,shape=2*val
    r, c = row[start_step + n_v:start_step + n_v + n_t], col[start_step + n_v:start_step + n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)   # 将20%到40%划分为训练集
    r, c = row[start_step + n_v + n_t:start_step + n_a], col[start_step + n_v + n_t:start_step + n_a]   # 划分训练集
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    diag = torch.diag(neg_adj_mask)
    reshape_diag = torch.diag_embed(diag)
    neg_adj_mask = (neg_adj_mask-reshape_diag).to(torch.bool)
    neg_adj_mask[row_original, col_original] = 0                              # 将positive edge置为false
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()  # as_tuple得填，转置也得填
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    neg_adj_mask[neg_row, neg_col] = 0                      # 把训练集和测试集的负样本也置为0
    data.train_neg_adj_mask = neg_adj_mask
    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)
    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def direction_blind(data, fold, seed, val_ratio=config.val_ratio, test_ratio=config.test_ratio):

    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    row_original, col_original = data.edge_index
    data.edge_index = None
    n_v = int(math.floor(val_ratio * row_original.size(0)))
    n_t = int(math.floor(test_ratio * row_original.size(0)))
    n_a = int(math.floor(row_original.size(0)))

    # Positive edges.
    perm = torch.randperm(row_original.size(0))
    start_step = int(fold / config.fold * perm.size().numel())
    perm_repeat = torch.cat([perm, perm], dim=0)
    row, col = row_original[perm_repeat], col_original[perm_repeat]
    r, c = row[start_step:start_step + n_v], col[start_step:start_step + n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_neg_edge_index = torch.stack([c, r], dim=0)
    r, c = row[start_step + n_v:start_step + n_v + n_t], col[start_step + n_v:start_step + n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_neg_edge_index = torch.stack([c, r], dim=0)
    r, c = row[start_step + n_v + n_t:start_step + n_a], col[start_step + n_v + n_t:start_step + n_a]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # data.train_neg_edge_index = torch.stack([c, r], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    diag = torch.diag(neg_adj_mask)
    reshape_diag = torch.diag_embed(diag)
    neg_adj_mask = (neg_adj_mask-reshape_diag).to(torch.bool)
    neg_adj_mask[row_original, col_original] = 0
    neg_adj_mask[col_original[:n_v + n_t], row_original[:n_v + n_t]] = 0
    data.train_neg_adj_mask = neg_adj_mask

    return data
