import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import yaml
import pprint
from Dataset.HSIDataset import HSIDataset
from model.HSI_AE import HSI_AE
import datetime
import sys


def train_AE(cfg, pre_train_model):

    time = datetime.datetime.now()
    time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_path = cfg['MODEL']['AE']['MODEL_PATH'] + '/' + 'modelAE' + time + '.pth'

    batch_size = cfg['TRAIN']['AE']['BATCH_SIZE']

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
        
    AE = HSI_AE(cfg['MODEL']['AE']['LAYER_CHANNEL'], 
                len(cfg['MODEL']['AE']['LAYER_CHANNEL']),
                cfg['MODEL']['AE']['MAXPOOL_AFTER'],
                cfg['MODEL']['AE']['STRIDE_BEFORE']).cuda()

    print(AE)

    if pre_train_model != None:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'Load From exisiting model: ' + pre_train_model)
        AE.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH'] + '/' + pre_train_model))
    
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

    optimizer = optim.Adam(AE.parameters(), lr=cfg['OPTIMIZER']['lr'])
    loss_fn = nn.MSELoss(reduction='mean')

    train_losses = []
    train_counter = []

    for epoch in range(cfg['TRAIN']['AE']['EPOCH']):
        AE.train()
        for batch_idx, data in enumerate(train_loader):
            hsi_img = data[0]
            optimizer.zero_grad()
            hsi_img = hsi_img.cuda()
            _, output = AE(hsi_img)
            output_size = output.shape
            # hsi_img = hsi_img[:,:,:output.shape[2],:output.shape[3]]
            loss = loss_fn(output, hsi_img)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item()/batch_size) # item() is to get the value of the tensor directly
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: [{batch_idx*len(hsi_img)}/{len(train_loader.dataset)}] Loss: {loss.item()/batch_size}')
        val_loss = validation(epoch, cfg, AE, test_dataset)
        if best_loss == -1 or best_loss > val_loss:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: Better loss -- Save Model')
            best_loss = val_loss
            best_loss_epoch = epoch
            torch.save(AE.state_dict(), model_path)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Best Loss: {best_loss} @ Epoch {best_loss_epoch}')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' Best Model Saved at: ' + model_path)
    

    

def validation(epoch, cfg, model, dataset):
    batch_size = cfg['TEST']['BATCH_SIZE']

    model.eval()
    
    test_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False)

    loss_fn = nn.MSELoss(reduction='mean')
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data[0]
            image = image.cuda()
            _, output = model(image)
            test_loss += loss_fn(output, image).item()
            # image = F.pad(image, [0, output.shape[2], 0, output.shape[3]])
        test_loss /= len(test_loader.dataset)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: Test result on the model: Avg Loss is {test_loss}')
    return test_loss
    


if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    args = sys.argv
    train_AE(cfg, args[1] if len(args)>1 else None)