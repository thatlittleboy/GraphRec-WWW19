import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
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
            Embedding layer for the item node ids

        r2e (nn.Embedding):
            Embedding layer for the rating scores

        u2e (nn.Embedding):
            Embedding layer for the user node ids

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
        nodes (list):
            List of "self" nodes undergoing the neighbour aggregation in the user-item graph.

        history_uv (list of list):
            history_uv[i] is a list of neighbours of nodes[i].
            Must have the same length as `nodes`.

        history_r (list of list):
            history_r[i] is a list of rating values, where its jth element corresponds to the edge
            triplet (nodes[i], history_r[i][j] history_uv[i][j]).
            Must have the same length as `nodes`.

        Returns
        -------
        to_feats:
            Either the item-space user latent factor (h^I_i) of user node u_i, or the item latent factor (z_j)
            of item node v_j. (???)
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

            # attention weights \alpha
            att_w = self.att(
                node1=o_history,
                u_rep=uv_rep,
                num_neighs=num_history_item,
            )

            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history

        to_feats = embed_matrix
        return to_feats
