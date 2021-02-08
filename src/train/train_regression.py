import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pprint
from Dataset.HSIDataset import HSIDataset
from model.HSI_Regre import HSI_Regre
from model.Spec_AE import Spec_AE
import datastime
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

    dataset = DAtaset.SpecDataset(cfg=cfg, train=True)

    test_dataset = Dataset.SpecDataset(cfg=cfg, train=False)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(Regre.parameters(), lr=cfg['OPTIMIZER']['lr'])
    loss_fn = nn.M


