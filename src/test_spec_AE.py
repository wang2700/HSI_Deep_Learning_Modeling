import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import yaml
import pprint
from Dataset.SpecDataset import SpecDataset
from model.Spec_AE import Spec_AE
import matplotlib.pyplot as plt
import matplotlib
import Dataset.HSI_Analysis as HSI_Analysis
import numpy as np
import cv2
import sys

def test(cfg, model_name):
    batch_size = cfg['TEST']['BATCH_SIZE']

    AE = Spec_AE(cfg['MODEL']['SPEC_AE']['N_WAVELENGTH'], 
                cfg['MODEL']['SPEC_AE']['N_LATENT'],
                cfg['MODEL']['SPEC_AE']['N_STAGES'])
    AE.load_state_dict(torch.load(cfg['MODEL']['SPEC_AE']['MODEL_PATH'] + '/' + model_name))
    AE = AE.cuda()
    AE.eval()

    dataset = SpecDataset(cfg=cfg, train=False)
    
    test_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=1,
                                                shuffle=False)

    loss_fn = nn.MSELoss(reduction='mean')
    test_loss = 0
    print("Start Testing AE")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            specs = torch.flatten(data[0], start_dim=0, end_dim=1)
            specs = specs.cuda()
            _, output = AE(specs)
            test_loss += loss_fn(output, specs).item()
            location = torch.randint(low=0, high=specs.shape[0], size=(5,))
            print("Check at location: ", location)
            evaluate(cfg, output, specs, location)
        
        test_loss /= len(test_loader.dataset)
        print(f'Test result on the model: Avg Loss is {test_loss}')

def evaluate(cfg, output, original, location):
    output = output.detach().cpu().numpy()
    original = original.cpu().numpy()
    fig, ax = plt.subplots(len(location), 1)
    fig.set_size_inches(8, 4*len(location))
    wv = HSI_Analysis.getWvs(range=np.arange(cfg['DATASET']['N_WAVELENGTH']), paraPst2Wv=cfg['HSI']['PST2WV'])

    for i in range(len(location)):
        ax[i].plot(wv, output[i,:], label='output')
        ax[i].plot(wv, original[i,:], label='original')
        ax[i].legend()
        if (i == (len(location)- 1)):
            ax[i].set_xlabel('wavelength')
        ax[i].set_ylabel('intensity')
        ax[i].set_ylim([0, 1.6])
    plt.show()

    
def analysis(cfg, image):
    specsAlongLeaf = np.zeros(image.shape[1:3])
    ndviImg = HSI_Analysis.getNDVIHeatmap(image, cfg['HSI']['WV2PST'])
    imgseg = HSI_Analysis.segmentation(image, ndviImg, cfg['HSI']['NDVI_THRESH'])
    imgseg = HSI_Analysis.removeRightRegion(imgseg)
    ndviImg = HSI_Analysis.getNDVIHeatmap(imgseg, cfg['HSI']['WV2PST'])
    specsAlongLeaf = HSI_Analysis.getMeanSpecAlongLeaf(imgseg)
    specsMean = HSI_Analysis.getMeanSpecWholeLeaf(specsAlongLeaf)
    wv = HSI_Analysis.getWvs(range=np.arange(image.shape[2]), paraPst2Wv=cfg['HSI']['PST2WV'])
    return ndviImg, specsAlongLeaf, specsMean, wv



if __name__ == "__main__":
    args = sys.argv
    cfg = yaml.load(open('/media/jerrynas/Research/LeafSpec/Models/HSI Deep Learning Model/Jan2020/' + args[2]), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
    for gui in gui_env:
        try:
            print("testing", gui)
            matplotlib.use(gui,warn=False, force=True)
            from matplotlib import pyplot as plt
            break
        except:
            continue
    print("Using:",matplotlib.get_backend())

    test(cfg, args[1])