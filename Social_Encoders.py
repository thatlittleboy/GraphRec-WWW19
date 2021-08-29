import torch
import torch.nn as nn
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        """
        Parameters
        ----------
        features (nn.Embedding):
            Embedding layer of

        embed_dim (int):
            number of embedding dimensions for the social

        social_adj_lists
        """
        super(Social_Encoder, self).__init__()

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model is not None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):

        # ordered list of lists, each list being the node's neighbours
        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network
        # neigh_feats has dimensions (num_nodes, embed_dim)

        self_feats = self.features(
            torch.LongTensor(nodes.cpu().numpy())
        ).to(self.device)
        self_feats = self_feats.t()
        # self_feats has dimensions (num_nodes, embed_dim)

        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)  # (num_nodes, 2*embed_dim)
        combined = F.relu(self.linear1(combined))  # (num_nodes, embed_dim)

        return combined
