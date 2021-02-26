import torch
from distilled_ctr.layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x, sigmoid_output=True):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not x.is_cuda:
            x = torch.LongTensor(x).to(device)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        out = torch.sigmoid(x.squeeze(1)) if sigmoid_output else x
        return out
