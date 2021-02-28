import torch
import tqdm
import time
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import distilled_ctr.config as config
from ptflops import get_model_complexity_info

from distilled_ctr.dataset.avazu import AvazuDataset
from distilled_ctr.dataset.criteo import CriteoDataset
from distilled_ctr.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from distilled_ctr.model.afi import AutomaticFeatureInteractionModel
from distilled_ctr.model.afm import AttentionalFactorizationMachineModel
from distilled_ctr.model.dcn import DeepCrossNetworkModel
from distilled_ctr.model.dfm import DeepFactorizationMachineModel
from distilled_ctr.model.ffm import FieldAwareFactorizationMachineModel
from distilled_ctr.model.fm import FactorizationMachineModel
from distilled_ctr.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from distilled_ctr.model.fnn import FactorizationSupportedNeuralNetworkModel
from distilled_ctr.model.hofm import HighOrderFactorizationMachineModel
from distilled_ctr.model.lr import LogisticRegressionModel
from distilled_ctr.model.ncf import NeuralCollaborativeFiltering
from distilled_ctr.model.nfm import NeuralFactorizationMachineModel
from distilled_ctr.model.pnn import ProductNeuralNetworkModel
from distilled_ctr.model.wd import WideAndDeepModel
from distilled_ctr.model.xdfm import ExtremeDeepFactorizationMachineModel
from distilled_ctr.model.afn import AdaptiveFactorizationNetwork
from distilled_ctr.model.dnn import DNNModel
from distilled_ctr.model.ensemble import EnsembleModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=32, mlp_dims=(32, 32), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=32, mlp_dims=(32,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=32, num_layers=3, mlp_dims=(32, 32), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=32, mlp_dims=(32, 32), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(32, 32), split_half=False, mlp_dims=(32, 32), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=32, attn_size=32, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    elif name == 'dnn':
        return DNNModel(field_dims, embed_dim=16)
    elif name.split('-')[1] == 'ensemble':
        model_names = ['dcn', 'dfm', 'xdfm']
        models = []
        for m_name in model_names:
            m = get_model(m_name, dataset)
            m.load_state_dict(torch.load(config.MODEL_DIR.format(model_name=m_name)))
            models.append(m.to(device))
        ensemble_type = name.split('-')[0]
        assert ensemble_type in ('avg', 'weighted', 'stacked')
        return EnsembleModel(models=models, ensemble_type=ensemble_type, field_dims=field_dims, embed_dim=16)
    else:
        raise ValueError('unknown model name: ' + name)




def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         args):
    def make_input(input_res):
        train_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.workers)
        for fields, _ in train_data_loader:
            x = fields.to(device)
            break
        assert x.shape == input_res, 'shape mismatch.'
        return {'x' : x}
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    model = get_model(model_name, dataset).to(device)
    x_shape = (batch_size,len(dataset.field_dims))
    macs, params = get_model_complexity_info(model, x_shape, as_strings=True,input_constructor=make_input,
                                        print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='avazu')
    #data/avazu/small
    parser.add_argument('--dataset_path', default='data/avazu/small',  help='data/criteo/train.txt, data/avazu/train, or ml-1m/ratings.dat')
   
    #parser.add_argument('--dataset_name', default='criteo')
    #parser.add_argument('--dataset_path', default='data/criteo/train.txt',  help='data/criteo/train.txt, data/avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='xdfm')
    parser.add_argument('--experiment', action='store', type=str, default='unnamed-experiment', help='name the experiment')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args)