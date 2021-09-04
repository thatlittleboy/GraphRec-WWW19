import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)

    def forward(self, node1, u_rep, num_neighs):
        """Returns the attention coefficients alpha (Eq. (6) in the paper, after
        softmax).

        Implemented as a simple 2-layer MLP (Attention Network). Eq. (5).

        Parameters
        ----------
        node1 (torch.Tensor):
            The embedding vectors of nodes v which are being aggregated over into
            target node u. Here, v are the neighbours of u, with size (num_neighs, embed_dim)

        u_rep (torch.Tensor):
            The embedding/hidden representation vector of _target_ node u. Size
            of (embed_dim,)

        num_neighs (int):
            Number of neighbours that the node u has

        Returns
        -------
        att (torch.Tensor):
            The attention weights to be applied on `node1`, of size (num_neighs, 1)
        """
        uv_reps = u_rep.repeat(num_neighs, 1)  # (num_neighs, embed_dim)
        x = torch.cat((node1, uv_reps), 1)  # (num_neighs, 2*embed_dim)

        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)

        x = self.att3(x)
        att = F.softmax(x, dim=0)

        return att
