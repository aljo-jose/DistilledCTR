import torch

from distilled_ctr.layer import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AttentionalFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.

    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)

    def forward(self, x, sigmoid_output=True):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not x.is_cuda:
            x = torch.LongTensor(x).to(device)
        x = self.linear(x) + self.afm(self.embedding(x))
        out = torch.sigmoid(x.squeeze(1)) if sigmoid_output else x
        return out
