import argparse
import datetime
import os
import pickle
import random
import time
from collections import defaultdict
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn import init
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator

"""
GraphRec: Graph Neural Networks for Social Recommendation.
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin.
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        """
        Parameters
        ----------
        enc_u (Social_Encoder):
            Social Encoder for the user nodes

        enc_v_history (UV_Encoder):
            Encoder for the item nodes (from the User Aggregation step)

        r2e (nn.Embedding):
            Embedding layer for converting discrete item rating to embedding vector
        """

        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e

        # Batch norm layers
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        """
        Parameters
        ----------
        nodes_u (list): Batch of user node ids
        nodes_v (list): Batch of item node ids. Must be same length as `nodes_u`.
        """
        embeds_u = self.enc_u(nodes_u)  # (batch_size, embed_dim)
        embeds_v = self.enc_v_history(nodes_v)  # (batch_size, embed_dim)

        # User Modeling (obtain h_i for each user node)
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        # Item Modeling (obtain z_j for each item node)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        # Rating Prediction
        x_uv = torch.cat((x_u, x_v), dim=1)  # Eq.(20), (batch_size, embed_dim)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.bn4(self.w_uv2(x)))  # Eq.(22), (batch_size, 16)
        x = F.dropout(x, training=self.training)

        scores = self.w_uv3(x)  # (batch_size, 1)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        """
        Parameters
        ----------
        nodes_u (list): batch of user node ids (integers)
        nodes_v (list): batch of item node ids (integers)
        labels_list: list of item ratings of the uv edge (floats)
        """
        scores = self.forward(nodes_u, nodes_v)  # (batch_size,)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # batch_nodes_u: (batch_size,) containing user node ids, no features.
        # batch nodes_v: (batch_size,) containing item node ids, no features.
        # labels_list: (batch_size,) containing rating values corresponding to the (u, v) edge
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()

        # main training step: (forward + calculate loss) of batch data
        loss = model.loss(
            batch_nodes_u.to(device),
            batch_nodes_v.to(device),
            labels_list.to(device),
        )

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0

    return 0  # useless?


def test(model, device, test_loader):
    model.eval()
    tmp_pred, target = [], []

    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)

    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/toy_dataset'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,\
        train_u, train_v, train_r, test_u, test_v, test_r,\
        social_adj_lists, ratings_list = pickle.load(data_file)
    """
    ------ description of toy dataset -------
    Adjacency lists
    * history_u_lists (dict):
        users' purchased history (item set in training set). I.e., maps from
        user node_id to a list of item node_ids.
        E.g. { 0:  [10, 35, 60], ... } means user 0 interacted with items 10, 35 and 60.

    * history_ur_lists (dict):
        users' rating score in the training set. I.e., maps from user node_id
        to a list of rating scores.
        E.g. { 0: [3, 7, 1], ... } means user 0 rated item 10 with a score 3; rated
        item 35 with a score 7.

    * history_v_lists (dict):
        user set (in training set) who have interacted with the item. I.e., maps
        from item node_id to a list of user node_ids.
        E.g. { 10: [0, ...], ... } means item 10 was interacted by user 0, etc.

    * history_vr_lists: (dict)
        items' rating score in the training set. I.e., maps from item node_id to
        a list of rating scores.
        E.g. { 10: [3, ...], ... } means item 10 was rated score 3 by user 0.

    * train_u (list):
        node_ids of the user nodes in the training set;
    * train_v (list):
        node_ids of the item nodes in the training set;
    * train_r (list):
        item rating values in the training_set; 14091 ratings
    NOTE: zip(train_u, train_r, train_v) -> [(u, r, v) edge triplets]

    * test_u (list):
        node_ids of the user nodes in the testing set;
    * test_v (list):
        node_ids of the item nodes in the testing set;
    * test_r (list):
        item rating values in the testing set; 3733 ratings
    NOTE: zip(test_u, test_r, test_v) -> [(u, r, v) edge triplets]

    * social_adj_lists (dict):
        user's connected neighborhoods via adjacency lists. I.e., map from each
        user node_id to a set of node_ids of those users' neighbours

    ratings_list (dict):
        rating value from 0.5 to 4.0 (8 opinion embeddings)
        { 0.5: 7, 1.0: 1, 1.5: 6, 2.0: 0, 2.5: 4, 3.0: 2, 3.5: 5, 4.0: 3 }
    """

    trainset = torch.utils.data.TensorDataset(
        torch.LongTensor(train_u),
        torch.LongTensor(train_v),
        torch.FloatTensor(train_r),
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testset = torch.utils.data.TensorDataset(
        torch.LongTensor(test_u),
        torch.LongTensor(test_v),
        torch.FloatTensor(test_r),
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)  # embedding layer for user nodes
    v2e = nn.Embedding(num_items, embed_dim).to(device)  # embedding layer for item nodes
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)  # embedding layer for item ratings (from user-item graph)

    # user feature (User Modeling)
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(
        u2e, embed_dim, history_u_lists, history_ur_lists,
        aggregator=agg_u_history, cuda=device, uv=True,
    )
    # neighbors
    agg_u_social = Social_Aggregator(
        lambda nodes: enc_u_history(nodes).t(), u2e,
        embed_dim, cuda=device,
    )
    enc_u = Social_Encoder(
        lambda nodes: enc_u_history(nodes).t(), embed_dim,
        social_adj_lists, agg_u_social,
        base_model=enc_u_history, cuda=device,
    )

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(
        v2e, embed_dim, history_v_lists, history_vr_lists,
        aggregator=agg_v_history, cuda=device, uv=False,
    )

    # model
    graphrec = GraphRec(
        enc_u=enc_u,
        enc_v_history=enc_v_history,
        r2e=r2e,
    ).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(graphrec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()
