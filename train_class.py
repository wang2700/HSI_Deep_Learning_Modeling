import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import yaml
import pprint
from Dataset.HSIDataset import HSIDataset
from model.HSI_AE import HSI_AE
from model.Treatment_Classifier import Treatment_Classifier
import datetime
import sys
import numpy as np
from test_AE import evaluate

def train(cfg, AE_model_name):
    
    time = datetime.datetime.now()
    time = time.strftime("%Y-%m-%d-%H-%M-%S")
    class_model_path = cfg['MODEL']['AE']['MODEL_PATH'] + '/' + 'modelClass' + time + '.pth'
    AE_model_path = cfg['MODEL']['AE']['MODEL_PATH'] + '/' + AE_model_name

    batch_size = cfg['TRAIN']['TREATMENT_CLASS']['BATCH_SIZE']
    epoch = cfg['TRAIN']['TREATMENT_CLASS']['EPOCH']

    best_loss = -1
    best_loss_epoch = -1

    transform_list = []
    if cfg['DATASET']['FLIP_HORIZONTAL']:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if cfg['DATASET']['FLIP_HORIZONTAL']:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if cfg['DATASET']['ROTATE_IMAGE']:
        transform_list.append(transforms.RandomRotation((-20, 20), expand=False))

    transform_list = transforms.Compose(transform_list)

    AE = HSI_AE(n_latent=cfg['MODEL']['AE']['N_LATENT'], n_wavelength=cfg['DATASET']['N_WAVELENGTH']).cuda()
    n_input = np.prod(cfg['MODEL']['AE']['OUTPUT_SIZE'])
    class_model = Treatment_Classifier(n_classes=cfg['MODEL']['TREATMENT_CLASS']['N_CLASSES'], 
                                        input_ch=cfg['MODEL']['AE']['N_LATENT']).cuda()
    
    if AE_model_name != None:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'Load From exisiting model: ' + AE_model_name)
        AE.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH'] + '/' + AE_model_name))

    AE.eval()
    
    dataset = HSIDataset(cfg=cfg,
                        root_dir=cfg['DATASET']['DATA_DIR'],
                        max_length=cfg['DATASET']['MAX_LENGTH'],
                        max_width=cfg['DATASET']['MAX_WIDTH'],
                        transform=transform_list,
                        train=True)

    test_dataset = HSIDataset(cfg=cfg,
                            root_dir=cfg['DATASET']['DATA_DIR'],
                            max_length=cfg['DATASET']['MAX_LENGTH'],
                            max_width=cfg['DATASET']['MAX_WIDTH'],
                            transform=None,
                            train=False)
    
    train_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=True)

    optimizer = optim.Adam(class_model.parameters(), lr=cfg['OPTIMIZER']['lr'])
    class_loss_fn = nn.BCELoss(reduction='sum')

    train_losses = []
    train_counter = []

    for epoch in range(cfg['TRAIN']['TREATMENT_CLASS']['EPOCH']):
        class_model.train()
        for batch_idx, data in enumerate(train_loader):
            hsi_img = data[0].cuda()
            gt = data[2].view(-1, 1).cuda()
            optimizer.zero_grad()
            features, output = AE(hsi_img)
            # for i in range(output.shape[0]):
            #     evaluate(cfg, output[i], hsi_img[i], data[1][i])
            pred = class_model(features)
            loss = class_loss_fn(pred, gt)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item()/batch_size)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: [{batch_idx*len(hsi_img)}/{len(train_loader.dataset)}] Loss: {loss.item()/batch_size}')
        val_loss = validation(epoch, cfg, AE, class_model,test_dataset)
        if best_loss == -1 or best_loss > val_loss:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: Better loss -- Save Model')
            best_loss = val_loss
            best_loss_epoch = epoch
            torch.save(class_model.state_dict(), class_model_path)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Best Loss: {best_loss} @ Epoch {best_loss_epoch}')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' Best Model Saved at: ' + class_model_path)

def validation(epoch, cfg, AE, class_model, dataset):
    batch_size = cfg['TEST']['BATCH_SIZE']
    AE.eval()
    class_model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False)
    loss_fn = nn.BCELoss(reduction='sum')
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data[0].cuda()
            gt = data[2].view(-1, 1).cuda()
            features, _ = AE(image)
            pred = class_model(features)
            test_loss += loss_fn(pred, gt).item()
        
        test_loss /= len(test_loader.dataset)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: Test result on the model: Avg Loss is {test_loss}')
    return test_loss

if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    args = sys.argv
    train(cfg, args[1] if len(args)>1 else None)