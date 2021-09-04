import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        """
        Parameters
        ----------
        v2e (nn.Embedding):
            Embedding layer for the item node ids, contains a `weight` tensor of size (num_items, embed_dim)

        r2e (nn.Embedding):
            Embedding layer for the rating scores, contains a `weight` tensor of size (num_ratings, embed_dim)

        u2e (nn.Embedding):
            Embedding layer for the user node ids, contains a `weight` tensor of size (num_users, embed_dim)

        embed_dim (int):
            The common dimension for all embedding vectors in this layer. For example, value used
            in the toy example is 64.

        cuda:

        uv (bool, Optional):
            Whether we're aggregating from item -> user (True, default; refers to Item Aggregation step
            from the paper) or from user -> item (False; refers to User Aggregation step from the paper)
        """
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.device = cuda
        self.embed_dim = embed_dim

        # the Embedding layers
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e

        # MLP for generating x_ia (Eq. (2)) OR f_jt (Eq. (15)) depending on uv
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)

        # MLP for ...
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):
        """
        Parameters
        ----------
        nodes (torch.Tensor):
            List of "self" nodes undergoing the neighbour aggregation in the user-item graph. Dimensions
            of (batch_size,)

        history_uv (list of list):
            history_uv[i] is a list of neighbours of nodes[i].
            Must have the same length as `nodes`.

        history_r (list of list):
            history_r[i] is a list of rating values, where its jth element corresponds to the edge
            triplet (nodes[i], history_r[i][j] history_uv[i][j]).
            Must have the same length as `nodes`.

        Returns
        -------
        to_feats (torch.Tensor):
            If uv=True, then this is the expression in braces in Eq. (4), i.e., the aggregated feature vector
            from neighbouring item nodes into each user node.
            If uv=False, then this is the expression in braces in Eq. (17), i.e., the aggregated feature vector
            from neighbouring user nodes into each item node.
            Dimensions in both cases are the same, (batch_size, embed_dims)
        """

        embed_matrix = torch.empty(
            len(history_uv), self.embed_dim,
            dtype=torch.float,
        ).to(self.device)

        # iterate through nodes'
        for i in range(len(history_uv)):
            history = history_uv[i]
            num_history_item = len(history)
            tmp_label = history_r[i]

            if self.uv:
                # user component
                e_uv = self.v2e.weight[history]  # e_uv == item embedding q_a from the paper
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]  # e_uv == user embedding p_t from the paper
                uv_rep = self.v2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]  # e_r == rating embedding vector

            # x == the opinion-aware interaction x_ia OR f_jt from the paper, depending on `uv`
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_weights = self.att(
                node1=o_history,
                u_rep=uv_rep,
                num_neighs=num_history_item,
            )
            # attention weights \alpha; dimensions (num_neighs, 1)

            att_history = torch.mm(o_history.t(), att_weights).t()  # (1, embed_dim)
            embed_matrix[i] = att_history

        to_feats = embed_matrix
        return to_feats
