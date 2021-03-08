import argparse
import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import distilled_ctr.config as config

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
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
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
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
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
            m.load_state_dict(torch.load(config.MODEL_DIR.format(model_name='criteo_'+m_name)))
            models.append(m.to(device))
        ensemble_type = name.split('-')[0]
        assert ensemble_type in ('avg', 'weighted', 'stacked')
        return EnsembleModel(models=models, ensemble_type=ensemble_type, field_dims=field_dims, embed_dim=16)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, model_name, save_dir, sample_x):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = f'{save_dir}/{model_name}.pt'
        self.jit_save_path = f'{save_dir}/{model_name}.jit.pt'
        self.model_name = model_name
        self.sample_x = sample_x

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            m = torch.jit.trace(model, self.sample_x)
            torch.jit.save(m, self.jit_save_path)
            print('better accuracy, model saved.')
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def entropy_loss_fn(outputs, targets):
    return nn.BCELoss()(outputs, targets)

def inverse_sigmoid(y):
    return torch.log(y/(1-y))

def distillation_loss_fn(student_scores, teacher_scores, targets, alpha=0.5):
    student_logits = inverse_sigmoid(student_scores)
    teacher_logits = inverse_sigmoid(teacher_scores)
    
    distill_loss = nn.MSELoss()(student_logits, teacher_logits) * (1. - alpha)
    student_loss = nn.BCELoss()(student_scores, targets) * alpha

    total_loss = distill_loss + student_loss
    return (distill_loss, student_loss, total_loss)

def train(args, teacher_model, student_model,  device, train_loader, optimizer, epoch):
    teacher_model.eval()
    student_model.train()

    tk0 = tqdm.tqdm(train_loader, smoothing=0, mininterval=1.0)
    for batch_idx, (data, targets) in enumerate(tk0):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        student_scores = student_model(data)
        with torch.no_grad(): # no gradients back to teacher.
            teacher_scores = teacher_model(data)
        
        distill_loss, student_loss, total_loss = distillation_loss_fn(
            student_scores, 
            teacher_scores, 
            targets.float(), 
            alpha = args.alpha) 
        total_loss.backward()
        optimizer.step()
        args.steps += 1

        args.writer.add_scalar('Loss/distilled', distill_loss.item(), args.steps)
        args.writer.add_scalar('Loss/student', student_loss.item(), args.steps)
        args.writer.add_scalar('Loss/total', total_loss.item(), args.steps)
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), total_loss.item()))
            # if args.dry_run:
            #     break


def test(args, model, device, test_loader):
    model.eval()
    targets, predicts = list(), list()
    total_loss = 0
    with torch.no_grad():
        for fields, target in tqdm.tqdm(test_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = entropy_loss_fn(y, target.float())
            total_loss += loss.item()
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    auc = roc_auc_score(targets, predicts)
    return auc, total_loss/len(test_loader)

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
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    torch.manual_seed(42)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=args.workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.workers)

    teacher_model = get_model(args.teacher, dataset)
    teacher_model_path = config.MODEL_DIR.format(model_name=args.teacher)
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    teacher_model = teacher_model.to(device)
    student_model = get_model(args.student, dataset).to(device)

    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sample_x = train_dataset.__getitem__(1) # sample input for jit save.
    sample_x = torch.LongTensor(sample_x[0]).to(device)
    early_stopper = EarlyStopper(num_trials=2, model_name=model_name, save_dir=save_dir, sample_x=sample_x)
    args.steps = 0
    for epoch_i in range(epoch):
        train(args, teacher_model, student_model,  device, train_data_loader, optimizer, epoch)
        auc, validation_loss = test(args, student_model, device, valid_data_loader)
        args.writer.add_scalar('auc/validation', auc, epoch_i+1)
        args.writer.add_scalar('loss/validation', validation_loss, epoch_i+1)
        print(f'epoch: {epoch_i}, validation: auc:, {auc}, validation loss:{validation_loss}')
        if not early_stopper.is_continuable(student_model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc,_ = test(args, student_model, device, test_data_loader)
    print(f'test auc: {auc}')
    args.writer.add_scalar('auc/test', auc, epoch_i+1)
    args.writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CTR Distillation Trainer')
    
    parser.add_argument('--dataset_name', default='avazu')
    #data/avazu/small
    parser.add_argument('--dataset_path', default='data/avazu/small',  help='data/criteo/train.txt, data/avazu/train, or ml-1m/ratings.dat')
    #parser.add_argument('--dataset_name', default='criteo')
    #parser.add_argument('--dataset_path', default='data/criteo/train.txt',  help='data/criteo/train.txt, data/avazu/train, or ml-1m/ratings.dat')
    
    parser.add_argument('--teacher', default='xdfm')
    parser.add_argument('--student', default='dnn')
    parser.add_argument('--experiment', action='store', type=str, default='unnamed-experiment', help='name the experiment')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--alpha', type=int, default=0.5)
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.writer = SummaryWriter(log_dir=f'logs/{args.experiment}-{round(time.time())}')
    args.model_name = args.teacher + '_' + args.student
    args.log_interval = 100
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