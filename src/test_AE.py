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
import matplotlib
import utils.HSI_Analysis as HSI_Analysis
import numpy as np
import cv2
import sys

def test(cfg, model_name):
    batch_size = cfg['TEST']['BATCH_SIZE']

    AE = HSI_AE(cfg['MODEL']['AE']['LAYER_CHANNEL'], 
                    len(cfg['MODEL']['AE']['LAYER_CHANNEL']),
                    cfg['MODEL']['AE']['MAXPOOL_AFTER'])
    AE.load_state_dict(torch.load(cfg['MODEL']['AE']['MODEL_PATH'] + '/' + model_name))
    AE = AE.cuda()
    AE.eval()

    dataset = HSIDataset(cfg=cfg,
                        root_dir=cfg['DATASET']['DATA_DIR'],
                        max_length=cfg['DATASET']['MAX_LENGTH'],
                        max_width=cfg['DATASET']['MAX_WIDTH'],
                        transform=None,
                        train=False)
    
    test_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False)

    loss_fn = nn.MSELoss(reduction='mean')
    test_loss = 0
    print("Start Testing AE")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data[0]
            image = image.cuda()
            _, output = AE(image)
            test_loss += loss_fn(output, image).item()
            for i in range(output.shape[0]):
                evaluate(cfg, output[i], image[i], data[1][i])
        
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
    args = sys.argv
    test(cfg, args[1])