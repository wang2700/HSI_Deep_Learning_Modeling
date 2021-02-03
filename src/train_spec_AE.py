import sys
sys.path.append('/home/Documents/Research/HSI_Deep_Learning_Modeling')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import yaml
import pprint
import Dataset.SpecDataset as Dataset
from model.Spec_AE import Spec_AE
import datetime
import sys


def train(cfg, pre_train_model):

    time = datetime.datetime.now()
    time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_path = cfg['MODEL']['SPEC_AE']['MODEL_PATH'] + '/' + time + '_modelSpecAE' + '.pth'

    batch_size = cfg['TRAIN']['SPEC_AE']['BATCH_SIZE']

    best_loss = -1
    best_loss_epoch = -1
    
    # transform_list = []
    # if cfg['DATASET']['FLIP_HORIZONTAL']:
    #     transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # if cfg['DATASET']['FLIP_HORIZONTAL']:
    #     transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    # if cfg['DATASET']['ROTATE_IMAGE']:
    #     transform_list.append(transforms.RandomRotation((-20, 20), expand=False))

    # transform_list = transforms.Compose(transform_list)
        
    AE = Spec_AE(cfg['MODEL']['SPEC_AE']['N_WAVELENGTH'], 
                cfg['MODEL']['SPEC_AE']['N_LATENT'],
                cfg['MODEL']['SPEC_AE']['N_STAGES']).cuda()

    print(AE)

    # if pre_train_model != None:
    #     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'Load From exisiting model: ' + pre_train_model)
    #     AE.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH'] + '/' + pre_train_model))
    
    dataset = Dataset.SpecDataset(cfg=cfg, train=True)

    test_dataset = Dataset.SpecDataset(cfg=cfg, train=False)
    
    train_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=1,
                                                shuffle=True)

    optimizer = optim.Adam(AE.parameters(), lr=cfg['OPTIMIZER']['lr'])
    loss_fn = nn.MSELoss(reduction='mean')

    for epoch in range(cfg['TRAIN']['SPEC_AE']['EPOCH']):
        AE.train()
        for batch_idx, data in enumerate(train_loader):
            # hsi_img = data[0]
            # specs = rearrageData(hsi_img)
            specs = torch.flatten(data[0], start_dim=0, end_dim=1)
            num_batch = int(specs.shape[0] / float(batch_size))
            train_loss_list = []
            for i in range(num_batch):
                optimizer.zero_grad()
                spec_batch = specs[i*batch_size:(i+1)*batch_size, :].cuda()
                _, output = AE(spec_batch)
                loss = loss_fn(output, spec_batch)
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: [{batch_idx}/{len(train_loader.dataset)}] Loss: {sum(train_loss_list)/len(train_loss_list)}')
        val_loss = validation(epoch, cfg, AE, test_dataset)
        if best_loss == -1 or best_loss > val_loss:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: Better loss -- Save Model')
            best_loss = val_loss
            best_loss_epoch = epoch
            torch.save(AE.state_dict(), model_path)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Best Loss: {best_loss} @ Epoch {best_loss_epoch}')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' Best Model Saved at: ' + model_path)

    # save yaml specified to this model
    with open(cfg['MODEL']['SPEC_AE']['MODEL_PATH'] + '/' + time + '_config' + '.yaml', 'w') as f:
        yaml.dump(cfg, f) 

    

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
            specs = torch.flatten(data[0], start_dim=0, end_dim=1)
            specs = specs.cuda()
            _, output = model(specs)
            test_loss += loss_fn(output, specs).item()
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Validation @ Epoch {epoch}: [{i}/{len(test_loader.dataset)}] Loss: {loss_fn(output, specs).item()}')
        test_loss /= len(test_loader.dataset)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f' Epoch {epoch}: Test result on the model: Avg Loss is {test_loss}')
    return test_loss
    
# def rearrageData(x):
#     x = torch.flatten(x, start_dim=2)
#     x = torch.flatten(x, start_dim=0, end_dim=1)
#     x = x.permute(1, 0)
#     return x

if __name__ == "__main__":
    args = sys.argv
    cfg = yaml.load(open('config/' + args[1]), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    train(cfg, None)