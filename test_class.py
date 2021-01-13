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
import matplotlib.pyplot as plt
import utils.HSI_Analysis as HSI_Analysis
import numpy as np
import cv2
import sys

def test(cfg, AE_model_name, class_model_name):
    batch_size = cfg['TEST']['BATCH_SIZE']

    print('Load AE Model')
    AE = HSI_AE(n_latent=cfg['MODEL']['AE']['N_LATENT'], n_wavelength=cfg['DATASET']['N_WAVELENGTH'])
    AE.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH'] + '/' + AE_model_name))
    AE = AE.cuda()
    AE.eval()

    print('Load Classification model')
    class_model = Treatment_Classifier(n_classes=cfg['MODEL']['TREATMENT_CLASS']['N_CLASSES'], 
                                        input_ch=cfg['MODEL']['AE']['N_LATENT'])
    class_model.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH'] + '/' + class_model_name))
    class_model = class_model.cuda()
    class_model.eval()

    dataset = HSIDataset(cfg=cfg,
                        root_dir=cfg['DATASET']['DATA_DIR'],
                        max_length=cfg['DATASET']['MAX_LENGTH'],
                        max_width=cfg['DATASET']['MAX_WIDTH'],
                        transform=None,
                        train=False)
    
    test_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False)

    loss_fn = nn.BCELoss(reduction='sum')
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data[0]
            image = image.cuda()
            gt = data[2].view(-1, 1).cuda()
            features, output = AE(image)
            pred = class_model(features)
            test_loss += loss_fn(pred, gt).item()
            print(pred, gt)
        
        test_loss /= len(test_loader.dataset)
        print(f'Test result on the model: Avg Loss is {test_loss}')

def evaluate(cfg, output, original, pst):
    output = output.permute((1,2,0)).detach().cpu().numpy()
    original = original.permute((1,2,0)).cpu().numpy()

    out_ndvi, out_specAlongLeaf, out_specWholeLeaf, out_wv = analysis(cfg, output)
    ori_ndvi, ori_specAlongLeaf, ori_specWholeLeaf, ori_wv = analysis(cfg, original)
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(12, 8)
    ax[0].imshow(ori_ndvi)
    ax[0].set_title('Original NDVI')
    ax[0].axis('off')
    ax[1].imshow(out_ndvi)
    ax[1].set_title('Output NDVI')
    ax[1].axis('off')
    plt.show()

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(12, 8)
    ax[0].plot(ori_wv, ori_specWholeLeaf)
    ax[0].set_title('Original Mean Spectrum')
    ax[1].plot(out_wv, out_specWholeLeaf)
    ax[1].set_title('Output Mean Spectrum')
    plt.show()

    # fig, ax = plt.subplots(1,2)
    # fig.set_size_inches(12, 8)
    # ax[0].imshow(ori_ndvi)
    # ax[0].set_title('Original NDVI')
    # ax[0].axis('off')
    # ax[1].imshow(out_ndvi)
    # ax[1].set_title('Output NDVI')
    # ax[1].axis('off')
    # plt.show()

    
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
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    pprint.pprint(cfg, indent=4)
    args = sys.argv
    test(cfg, args[1], args[2])