import torch
import torch.nn as nn
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        """
        Parameters
        ----------
        features (Callable):
            Transform input nodes (during `forward`) into the node features after _item_ aggregation.
            (batch_size,) -> (embed_dim, batch_size)

        embed_dim (int):
            number of embedding dimensions

        social_adj_lists (dict of lists):
            User's connected neighborhoods via adjacency lists. I.e., map from each
            user node_id to a set of node_ids of those users' neighbours

        aggregator ():
            Aggregator object that defines how the neighbours latent factors are aggregated together

        base_model ():
            ???

        cuda ():
        """
        super(Social_Encoder, self).__init__()

        self.device = cuda
        self.embed_dim = embed_dim

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model is not None:
            self.base_model = base_model

        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        """
        Parameters
        ----------
        nodes (torch.Tensor):
            list of node_ids of user nodes to generate encoding for (i.e., a batch of nodes)
            Size of (batch_size,)

        Returns
        -------
        combined (torch.Tensor):
            The user latent factors aggregated from each user nodes' neighbours. Dimension
            of (batch_size, embed_dim)
        """

        # Generate an ordered list of lists, w/ each list being the node's neighbours
        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network
        # `neigh_feats` is the expression in braces in Eq. (9), which is the aggregated neighbour features.
        # `neigh_feats` has dimensions (batch_size, embed_dim)

        # self-connection could be considered. (tho it doesn't seem to be in the paper)
        self_feats = self.features(
            torch.LongTensor(nodes.cpu().numpy())
        ).to(self.device)
        self_feats = self_feats.t()
        # `self_feats` has dimensions (batch_size, embed_dim)

        combined = torch.cat([self_feats, neigh_feats], dim=1)  # (batch_size, 2*embed_dim)

        # Implement Eq. (9)
        combined = F.relu(self.linear1(combined))  # (batch_size, embed_dim)
        return combined
