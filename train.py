import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import yaml
import pprint
from Dataset.HSIDataset import HSIDataset
from model.HSI_AE import HSI_AE



def train(cfg):
    
    batch_size = cfg['TRAIN']['BATCH_SIZE']
    
    transform_list = []
    if cfg['DATASET']['FLIP_HORIZONTAL']:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if cfg['DATASET']['FLIP_HORIZONTAL']:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    if cfg['DATASET']['ROTATE_IMAGE']:
        transform_list.append(transforms.RandomRotation((-20, 20), expand=False))

    transform_list = transforms.Compose(transform_list)
        
    AE = HSI_AE(n_latent=cfg['MODEL']['AE']['N_LATENT'], n_wavelength=cfg['DATASET']['N_WAVELENGTH']).cuda()
    
    dataset = HSIDataset(root_dir=cfg['DATASET']['DATA_DIR'],
                        max_length=cfg['DATASET']['MAX_LENGTH'],
                        transform=transform_list,
                        train=True)
    
    train_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=True)

    optimizer = optim.Adam(AE.parameters(), lr=cfg['OPTIMIZER']['lr'])
    loss_fn = nn.MSELoss(reduction='sum')

    train_losses = []
    train_counter = []

    for epoch in range(cfg['TRAIN']['EPOCH']):
        for batch_idx, data in enumerate(train_loader):
            hsi_img = data[0]
            optimizer.zero_grad()
            hsi_img = hsi_img.cuda()
            _, output = AE(hsi_img)
            loss = loss_fn(output, hsi_img)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item()/batch_size) # item() is to get the value of the tensor directly
            print(f'Epoch {epoch}: [{batch_idx*len(hsi_img)}/{len(train_loader.dataset)}] Loss: {loss.item()/batch_size}')

    torch.save(AE.state_dict(), cfg['MODEL']['AE']['MODEL_PATH'])

if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    train(cfg)