import os
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as F
import torch.optim as optim
import yaml
import pprint
import numpy as np
from Dataset.SpecDataset import SpecDataset
from model.HSI_Regre import HSI_Regre
from model.Spec_AE import Spec_AE
import datetime
import sys


def reconstruct(spec_feature, location, shape):
    reconstruct = torch.tensor(np.zeros(shape), dtype=torch.float32)
    k = 0
    for k in range(len(location[0])):
        reconstruct[:, location[0][k], location[1][k]] = spec_feature[k]
    return reconstruct


def extract_segemented_spec(data, i):
    mask = (data['mask'])[i]
    specs = (data['hsi_img'][i])[mask]
    specs = torch.tensor(specs, dtype=torch.float32)
    return specs


def reconstruct_img(cfg, features, location):
    reconstruct = torch.zeros(cfg['DATASET']['MAX_WIDTH'], cfg['DATASET']
                              ['MAX_LENGTH'], cfg['MODEL']['SPEC_AE']['N_LATENT'])
    for k in range(len(location[0])):
        i = location[0][k]
        j = location[1][k]
        reconstruct[i][j] = features[k]
    return reconstruct


def train_regre(cfg, model_path):
    AE = Spec_AE(cfg['MODEL']['SPEC_AE']['N_WAVELENGTH'],
                 cfg['MODEL']['SPEC_AE']['N_LATENT'],
                 cfg['MODEL']['SPEC_AE']['N_STAGES'])

    AE.load_state_dict(torch.load(os.path.join(model_path, 'specAE.pth')))
    AE = AE.cuda().eval()

    batch_size = cfg['TRAIN']['REGRESSION']['BATCH_SIZE']

    best_loss = -1
    best_loss = -1

    Regre = HSI_Regre(cfg['MODEL']['SPEC_AE']['N_LATENT'],
                      cfg['MODEL']['REGRESSION']['N_PRED'])
    Regre = Regre.cuda()

    print(Regre)

    dataset = SpecDataset(cfg=cfg, train=True)

    test_dataset = SpecDataset(cfg=cfg, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(Regre.parameters(), lr=cfg['OPTIMIZER']['lr'])
    loss_fn = nn.MSELoss(reduction='mean')

    train_losses = []

    for epoch in range(cfg['TRAIN']['REGRESSION']['EPOCH']):
        Regre.train()
        for batch_index, data in enumerate(train_loader):
            feat_map_shape = (data['hsi_img'].shape[0],
                              cfg['MODEL']['SPEC_AE']['N_LATENT'],
                              data['hsi_img'].shape[1],
                              data['hsi_img'].shape[2])
            feature_map = torch.zeros(feat_map_shape, device='cuda')
            for i in range(batch_size):
                spec = extract_segemented_spec(data, i)
                spec = spec.cuda()
                output, _ = AE(spec)
                del spec
                location = np.where(data['mask'][i])
                feature_map[i, :, :, :] = reconstruct(
                    output, location, feat_map_shape[1:])
            pred = Regre(feature_map)
            loss = loss_fn(pred, data['gt'][:, 0].reshape((4, 1)).cuda())
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                  f' Epoch {epoch}: [{batch_index}/{len(train_loader.dataset)}] Loss: {sum(train_losses)/len(train_losses)}')


if __name__ == "__main__":
    args = sys.argv
    cfg = yaml.load(open(
        '/home/jerry/Documents/Research/HSI_Deep_Learning_Modeling/config/' + args[1]), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=2)
    train_regre(cfg, 'test')
