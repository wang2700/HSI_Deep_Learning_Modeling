import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import yaml
import pprint
from Dataset.HSIDataset import HSIDataset
from model.HSI_AE import HSI_AE
import matplotlib.pyplot as plt

def test(cfg):
    batch_size = cfg['TEST']['BATCH_SIZE']

    AE = HSI_AE(n_latent=cfg['MODEL']['AE']['N_LATENT'], n_wavelength=cfg['DATASET']['N_WAVELENGTH'])
    AE.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH']))
    AE = AE.cuda()
    AE.eval()

    dataset = HSIDataset(root_dir=cfg['DATASET']['DATA_DIR'],
                        max_length=cfg['DATASET']['MAX_LENGTH'],
                        transform=None,
                        train=False)
    
    test_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False)

    loss_fn = nn.MSELoss(reduction='sum')
    test_loss = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.cuda()
            _, output = AE(image)
            test_loss += loss_fn(output, image).item()
            evaluate(output, image)
        
        test_loss /= len(test_loader.dataset)
        print(f'Test result on the model: Avg Loss is {test_loss}')

def evaluate(output, original):
    output = output.numpy()
    original = original.numpy()
    
     

if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    test(cfg)