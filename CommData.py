import os
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj

class DATA(object):

    def __init__(self):

        self.name = ["cora", "citeseer", "pubmed", "wiki"]

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_name = dataset
        if data_name in ["cora", "citeseer", "pubmed"]:
            path = os.path.split(os.path.realpath(__file__))[0] + "/commgnas/datasets/CITE/"
            dataset = Planetoid(path, dataset)
            data = dataset[0]

        else:
            data = self.load_wiki()

        edge_index = data.edge_index.to(device)
        x = data.x.to(device)
        y = data.y.to(device)

        index_list = [i for i in range(y.size(0))]

        # construct transductive node classification task mask
        if shuffle_flag:

            if not random_seed:
                random_seed = 123
            random.seed(random_seed)

            random.shuffle(index_list)

            if train_splits == None or val_splits == None:

                train_splits = self.count_(data.train_mask)
                val_splits = self.count_(data.val_mask)
                test_splits = self.count_(data.test_mask)

                idx_train = index_list[:train_splits]
                idx_val = index_list[train_splits:train_splits+val_splits]
                idx_test = index_list[train_splits+val_splits:train_splits+val_splits+test_splits]

            else:

                idx_train = index_list[:int(y.size(0) * train_splits)]
                idx_val = index_list[int(y.size(0) * train_splits):int(y.size(0) * train_splits) + int(y.size(0) * val_splits)]
                idx_test = index_list[int(y.size(0) * train_splits) + int(y.size(0) * val_splits):]
        else:

            if train_splits == None or val_splits == None:

                train_splits = self.count_(data.train_mask)
                val_splits = self.count_(data.val_mask)
                test_splits = self.count_(data.test_mask)

                idx_train = index_list[:train_splits]
                idx_val = index_list[train_splits:train_splits + val_splits]
                idx_test = index_list[train_splits + val_splits:train_splits + val_splits + test_splits]

            else:

                idx_train = index_list[:int(y.size(0) * train_splits)]
                idx_val = index_list[int(y.size(0) * train_splits):int(y.size(0) * train_splits) + int(y.size(0) * val_splits)]
                idx_test = index_list[int(y.size(0) * train_splits) + int(y.size(0) * val_splits):]

        self.train_mask = torch.tensor(self.sample_mask_(idx_train, y.size(0)), dtype=torch.bool)
        self.val_mask = torch.tensor(self.sample_mask_(idx_val, y.size(0)), dtype=torch.bool)
        self.test_mask = torch.tensor(self.sample_mask_(idx_test, y.size(0)), dtype=torch.bool)

        # construct random x1, x2, Edge1, Edge2

        drop_feature_rate1 = 0.3
        drop_feature_rate2 = 0.4
        drop_edge_rate1 = 0.2
        drop_edge_rate2 = 0.4

        x1 = self.drop_feature(x, drop_feature_rate1)
        x2 = self.drop_feature(x, drop_feature_rate2)

        edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate1)[0]
        edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate2)[0]

        # combine x1, x2 to a whole graph dataset
        x1_node_nums = x1.shape[0]

        edge_index_2[0] = edge_index_2[0] + x1_node_nums
        edge_index_2[1] = edge_index_2[1] + x1_node_nums

        # combine x1 and x2
        x_ = torch.cat([x1, x2], dim=0)

        # combine edge_index_1 and edge_index_2:
        edge_index_ = torch.cat([edge_index_1, edge_index_2], dim=-1)

        # Auto-GNAS input required attribution

        self.test_y = y

        self.train_x = x_
        self.val_x = x_

        self.test_x = x

        self.train_edge_index = edge_index_
        self.val_edge_index = edge_index_

        self.test_edge_index = edge_index

        self.num_features = data.num_features
        self.num_labels = y.max().item() + 1
        self.data_name = data_name
        self.node_nums = x1_node_nums

    def sample_mask_(self, idx, l):
        """ create mask """
        mask = np.zeros(l)
        for index in idx:
            mask[index] = 1
        return np.array(mask, dtype=np.int32)

    def count_(self, mask):
        true_num = 0
        for i in mask:
            if i:
                true_num += 1
        return true_num

    def drop_feature(self, x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def load_wiki(self):
        # load a dummy dataset to return the data in the same format as
        # those available in pytorch geometric
        path = os.path.split(os.path.realpath(__file__))[0] + "/commgnas/datasets/CITE/"
        dataset = Planetoid(path, "cora")
        data = dataset[0]

        wiki_path = os.path.split(os.path.realpath(__file__))[0]

        # replace with actual data from Wiki
        features = 0 * torch.FloatTensor(2405, 4973)
        adj = 0 * torch.LongTensor(2, 17981)
        labels = 0 * torch.LongTensor(2405)

        with open(wiki_path + '/wiki/graph.txt', 'r') as f:
            i = 0
            for line in f:
                temp_list = line.split()
                adj[0, i] = int(temp_list[0])
                adj[1, i] = int(temp_list[1])
                i += 1

        with open(wiki_path + '/wiki/tfidf.txt', 'r') as f:
            i = 0
            for line in f:
                temp_list = line.split()
                u = int(temp_list[0])
                v = int(temp_list[1])
                features[u, v] = float(temp_list[2])
                i += 1

        with open(wiki_path + '/wiki/group.txt', 'r') as f:
            i = 0
            for line in f:
                temp_list = line.split()
                node = int(temp_list[0])
                label = int(temp_list[1])
                # labels 12 and 14 are missing in data. Rename 18 and 19 to 12 and 14
                if label == 18:
                    label = 12
                if label == 19:
                    label = 14
                labels[node] = label - 1
                i += 1

        data.x = features
        data.y = labels
        data.edge_index = adj
        data.num_features = features.shape[1]
        return data

if __name__=="__main__":

    graph = DATA()
    graph.get_data("cora", shuffle_flag=False)
    pass