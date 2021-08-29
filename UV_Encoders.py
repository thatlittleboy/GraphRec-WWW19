import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class UV_Encoder(nn.Module):
    """Encoder for the user-item graph"""

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="cpu", uv=True):
        """
        Parameters
        ----------
        features ():

        embed_dim (int):

        history_uv_lists (dict of lists):
            Mapping from { user node id -> list of item node ids } when uv = True, or mapping from
            { item node id -> list of user node ids } when uv = False.

        history_r_lists (dict of lists):
            Mapping from { user node id -> list of item ratings } when uv = True, or mapping from
            { item node id -> list of item ratings } when uv = False.

        aggregator ():

        cuda ():

        uv (bool, Optional):
            Whether we're aggregating from item -> user (True, default; refers to Item Aggregation step
            from the paper) or from user -> item (False; refers to User Aggregation step from the paper)
        """
        super(UV_Encoder, self).__init__()

        self.uv = uv
        self.device = cuda
        self.embed_dim = embed_dim

        self.features = features
        self.history_uv_lists = history_uv_lists
        self.history_r_lists = history_r_lists
        self.aggregator = aggregator

        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        """
        Parameters
        ----------
        nodes (list):
            list of node_ids of nodes to generate encoding for

        Returns
        -------
        combined (torch.Tensor):
            dimension (num_nodes, embed_dim)
        """
        tmp_history_uv = []  # will be a list of lists, each list is the list of neighbours in the user-item graph for `node`
        tmp_history_r = []  # will be a list of lists, each list is the list of item ratings corresponding to the uv edge
        for node in nodes:
            tmp_history_uv.append(self.history_uv_lists[int(node)])
            tmp_history_r.append(self.history_r_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r)  # user-item network
        # neigh_feats has dimensions (num_nodes, embed_dim)

        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
