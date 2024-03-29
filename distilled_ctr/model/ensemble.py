import torch
import torch.nn as nn
import torch.nn.functional as F
from distilled_ctr.layer import FeaturesEmbedding

class EnsembleModel(nn.Module):
    def __init__(self, models,  ensemble_type, field_dims, embed_dim):
        super(EnsembleModel,self).__init__()
        self.ensemble_type = ensemble_type
        self.models = models
        self.num_models = len(self.models)
        if self.ensemble_type == 'weighted':
            #self.ensemble_weights = nn.Linear(self.num_models, 1, bias=False)
            self.weight_embed = nn.Embedding(self.num_models, 1)
        elif self.ensemble_type == 'stacked':
            self.stacked_out_lin = nn.Linear(370, 1)
        elif self.ensemble_type == 'gated':
            self.gated_lin1 = nn.Linear(len(field_dims) * embed_dim, self.num_models)
            self.gated_lin2 = nn.Linear(self.num_models, 1)

    def forward(self, x):
        ensemble_out = []
        with torch.no_grad(): # make sure ensembled models are not changed.
            for model in self.models:
                p = model(x.clone(), sigmoid_output=False)
                ensemble_out.append(p)

        if self.ensemble_type == 'avg':
            x = torch.stack(ensemble_out,dim=1).squeeze()
            out = torch.mean(x, dim=1, keepdim=True)
        elif self.ensemble_type == 'weighted':
            weights = self.weight_embed(torch.tensor(list(range(self.num_models)), dtype=torch.long, device=x.device)).view(1,self.num_models)
            smoothed_weights = nn.Softmax(dim=1)(weights)
            x = torch.stack(ensemble_out,dim=1).squeeze()
            x = x * smoothed_weights
            out = x.sum(dim=1,keepdim=True)
        elif self.ensemble_type == 'stacked': # implement output type.
            x = torch.cat((ensemble_out), dim=1)
            out = self.stacked_out_lin(x)
        elif self.ensemble_type == 'gated':
            x = self.gated_lin1(x)
            x = F.relu(x)
            x = x * torch.tensor(ensemble_out).reshape(x.shape)
            x = self.gated_lin2(x)
        else:
            raise ValueError('unexpected ensembling type ', self.ensemble_type)

        return torch.sigmoid(out.squeeze(1))