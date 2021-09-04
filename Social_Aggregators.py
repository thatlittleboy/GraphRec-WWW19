import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from Attention import Attention


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    Implements `Aggre_{neighbours}` in Eq. (7).
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        """
        Parameters
        ----------
        features (Callable):
            Transform input nodes (during `forward`) into the node features after _item_ aggregation.
            (batch_size,) -> (embed_dim, batch_size)

        u2e (nn.Embedding):
            Embedding layer for the user node ids, contains a `weight` tensor of size (num_users, embed_dim)

        embed_dim (int):
            The common dimension for all embedding vectors in this layer. For example, value used
            in the toy example is 64.

        cuda ():
        """
        super(Social_Aggregator, self).__init__()

        self.device = cuda
        self.embed_dim = embed_dim

        self.features = features
        self.u2e = u2e
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        """
        Parameters
        ----------
        nodes (torch.Tensor):
            list of node_ids of user nodes to generate encoding for (i.e., a batch of nodes)
            Size of (batch_size,)

        to_neighs (list of lists):
            Ordered list (same order as `nodes`) of each nodes' neighbours

        Returns
        -------
        to_feats (torch.Tensor):
            The features after aggregation from v->u, i.e., the expression in braces of Eq. (9).
            Size (batch_size, embed_dim)
        """
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i, self_node in enumerate(nodes):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)

            # fast: extract out the embedding vector for the neighbour nodes from `u2e`
            e_u = self.u2e.weight[list(tmp_adj)]

            # slow: item-space user latent factor (item aggregation) -> h_o^I in Eq. (7)-(9)
            # feature_neighbors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            # e_u = feature_neighbors.t()
            # `e_u` has size (num_neighs, embed_dim)

            u_rep = self.u2e.weight[self_node]
            # `u_rep` is the embedding vector for the self node, and has size (embed_dim,)

            att_weights = self.att(e_u, u_rep, num_neighs)  # (num_neighs, 1)
            att_history = torch.mm(e_u.t(), att_weights).t()  # (1, embed_dim)
            embed_matrix[i] = att_history

        to_feats = embed_matrix
        return to_feats
