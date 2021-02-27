import torch
import torch.nn as nn
import torch.nn.functional as F
from distilled_ctr.layer import FeaturesEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DNNModel(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(DNNModel,self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.lin1 = nn.Linear(self.embed_output_dim, 4)
        self.lin2 = nn.Linear(4,4)
        self.lin3 = nn.Linear(4,1)
        #self.drop_out_1 = torch.nn.Dropout(p=dropout)
        
    def forward(self, x, sigmoid_output=True):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not x.is_cuda:
            x = torch.LongTensor(x).to(device)
        embed_x = self.embedding(x)
        embed_x = embed_x.view(-1, self.embed_output_dim)
        x = self.lin1(embed_x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        out = torch.sigmoid(x.squeeze(1)) if sigmoid_output else x
        return out