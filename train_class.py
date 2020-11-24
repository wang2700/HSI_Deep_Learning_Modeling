import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import pprint
from Dataset.HSIDataset import HSIDataset
from model.HSI_AE import HSI_AE
import datetime
import sys

def train(cfg, AE_model_name):
    
    time = datetime.datetime.now()
    time = time.strftime("%Y-%m-%d-%H-%M-%S")
    class_model_path = cfg['MODEL']['AE']['MODEL_PATH'] + '/' + 'modelClass' + time + '.pth'
    AE_model_path = cfg['MODEL']['AE']['MODEL_PATH'] + '/' + AE_model_name

    batch_size = cfg['TRAIN']['TREATMENT_CLASS']['BATCH_SIZE']
    epoch = cfg['TRAIN']['TREATMENT_CLASS']['EPOCH']

    best_loss = -1
    best_loss_epoch = -1

if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    args = sys.argv
    train(cfg, args[1] if len(args)>1 else None)