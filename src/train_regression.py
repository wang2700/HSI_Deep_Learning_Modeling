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
from math import floor


def reconstruct(spec_feature, location, shape):
    reconstruct = torch.tensor(np.zeros(shape), dtype=torch.float32)
    k = 0
    for k in range(len(location[0])):
        reconstruct[:, location[0][k], location[1][k]] = spec_feature[k]
    return reconstruct.detach()


def extract_segemented_spec(data, i):
    mask = (data['mask'])[i]
    specs = (data['hsi_img'][i])[mask]
    # specs = torch.tensor(specs, dtype=torch.float32)
    return specs


# def reconstruct_img(cfg, features, location):
#     reconstruct = torch.zeros(cfg['DATASET']['MAX_WIDTH'], cfg['DATASET']
#                               ['MAX_LENGTH'], cfg['MODEL']['SPEC_AE']['N_LATENT'])
#     for k in range(len(location[0])):
#         i = location[0][k]
#         j = location[1][k]
#         reconstruct[i][j] = features[k]
#     return reconstruct


def train_regre(cfg, model_path, train_dataset, test_dataset):

    TAG = 'REGRE'

    AE = Spec_AE(cfg['MODEL']['SPEC_AE']['N_WAVELENGTH'],
                 cfg['MODEL']['SPEC_AE']['N_LATENT'],
                 cfg['MODEL']['SPEC_AE']['N_STAGES'])

    AE.load_state_dict(torch.load(os.path.join(model_path, 'specAE.pth')))
    AE = AE.cuda().eval()

    batch_size = cfg['TRAIN']['REGRESSION']['BATCH_SIZE']

    model_path = os.path.join(model_path, 'regre.pth')

    best_loss = -1
    best_loss_epoch = -1

    Regre = HSI_Regre(cfg['MODEL']['SPEC_AE']['N_LATENT'],
                      cfg['MODEL']['REGRESSION']['N_PRED'])
    Regre = Regre.cuda()

    print(Regre)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

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
            for i in range(data['hsi_img'].shape[0]):
                spec = extract_segemented_spec(data, i)
                spec = spec.cuda()
                output, _ = AE(spec)
                del spec
                location = np.where(data['mask'][i])
                feature_map[i, :, :, :] = reconstruct(
                    output, location, feat_map_shape[1:])
            pred = Regre(feature_map)
            loss = loss_fn(pred, data['gt'][:, 2].reshape((-1, 1)).cuda())
            loss.backward()
            optimizer.step()
            train_losses.append(loss)

            # delete variable
            del data
            del feature_map
            del pred
            del loss
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + TAG + ' -'
                  f' Epoch {epoch}: [{batch_index}/{floor(len(train_loader.dataset) / float(batch_size))}] Loss: {sum(train_losses)/len(train_losses)}')
        val_loss = validation(epoch, cfg, AE, Regre, test_dataset)

        if best_loss == -1 or best_loss > val_loss:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + TAG + ' -'
                  f' Epoch {epoch}: Better loss -- Save Model')
            best_loss = val_loss
            best_loss_epoch = epoch
            torch.save(Regre.state_dict(), model_path)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + TAG + ' -'
          f' Best Loss: {best_loss} @ Epoch {best_loss_epoch}')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + TAG + ' -'
          ' Best Model Saved at: ' + model_path)


def validation(epoch, cfg, AE, Regre, dataset):
    batch_size = cfg['TEST']['REGRESSION']['BATCH_SIZE']

    AE.eval()
    Regre.eval()

    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    loss_fn = nn.MSELoss(reduction='mean')
    test_loss = 0
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):
            feat_map_shape = (data['hsi_img'].shape[0],
                              cfg['MODEL']['SPEC_AE']['N_LATENT'],
                              data['hsi_img'].shape[1],
                              data['hsi_img'].shape[2])
            feature_map = torch.zeros(feat_map_shape, device='cuda')
            for i in range(data['hsi_img'].shape[0]):
                spec = extract_segemented_spec(data, i)
                spec = spec.cuda()
                output, _ = AE(spec)
                del spec
                location = np.where(data['mask'][i])
                feature_map[i, :, :, :] = reconstruct(
                    output, location, feat_map_shape[1:])
            pred = Regre(feature_map)
            test_loss += torch.sum(loss_fn(pred,
                                           data['gt'][:, 0].reshape((-1, 1)).cuda()))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
              f' Epoch {epoch}: Test result on the model: Avg Loss is {test_loss/len(dataset)}')
    return test_loss/len(dataset)


if __name__ == "__main__":
    args = sys.argv
    cfg = yaml.load(open(
        '/home/jerry/Documents/Research/HSI_Deep_Learning_Modeling/config/' + args[1]), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=2)
    train_regre(cfg, 'test')
