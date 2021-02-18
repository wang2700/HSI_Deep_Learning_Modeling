import torch
import torch.nn as nn
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

def train_regre(cfg, folder_name):
    AE = Spec_AE(cfg['MODEL']['SPEC_AE']['N_WAVELENGTH'],
                cfg['MODEL']['SPEC_AE']['N_LATENT'],
                cfg['MODEL']['SPEC_AE']['N_STAGES'])

    AE.load_state_dict(torch.load(cfg['MODEL']['SPEC_AE']['MODEL_PATH'] + '/' + folder_name + '/Spec_AE.pth'))
    AE = AE.cuda().eval()

    batch_size = cfg['TRAIN']['REGRESSION']['BATCH_SIZE']

    best_loss = -1
    best_loss = -1

    Regre = HSI_Regre(cfg['MODEL']['Spec_AE']['N_Latent'], cfg['MODEL']['REGRESSION']['N_PRED'])
    Regre = Regre.cuda()

    print(Regre)

    dataset = SpecDataset(cfg=cfg, train=True)

    test_dataset = SpecDataset(cfg=cfg, train=False)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(Regre.parameters(), lr=cfg['OPTIMIZER']['lr'])
    loss_fn = nn.MSELoss(reduction='mean')

    train_losses = []

    for epoch in range(cfg['TRAIN']['REGRESSION']['EPOCH']):
        Regre.train()
        for batch_index, data in enumerate(train_loader):
            for i in range(batch_index):
                spec = (data[0])[i,:,:]
                spec = spec.cuda()
                _, ouput = AE(spec)
                
def reconstruct(spec_feature, location, shape):
    reconstruct = torch.tensor(np.zeros(shape), dtype=torch.float32)
    k = 0
    for k in range(len(location[0])):
        reconstruct[location[0][k]][location[1][k]] = spec_feature[k]
    return reconstruct