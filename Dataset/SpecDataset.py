import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from .files import Files
from .HSI_Analysis import caliColor, getMask, getNDVIHeatmap
import csv
import yaml

class SpecDataset(Dataset):
    def __init__(self, cfg, train=True):
        self.cfg = cfg
        self.root_dir = cfg['DATASET']['DATA_DIR']
        # self.transform = transform
        self.max_width = cfg['DATASET']['MAX_WIDTH']
        mfiles = Files(self.root_dir, ['*heatmap.npy'])
        trainFiles = []
        testFiles = []
        for i, filename in enumerate(mfiles.filesNoExt):
            if (i % 4 == 0):
                testFiles.append(filename)
            else:
                trainFiles.append(filename)
        # print(trainFiles)
        # print(testFiles)
        self.rawNameList = trainFiles if train else testFiles
        # self.rawNameList = mfiles.filesNoExt
        self.rawNameList = [n[:-8] for n in self.rawNameList]
        # self.max_length = max_length

    def __len__(self): 
        return len(self.rawNameList)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #load images
        image = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '.npy')
        whiteRef = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '_whRef.npy')
        pst = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '_imgPst.npy')
        #white reference calibration
        image = caliColor(image, whiteRef)
        #get mask
        ndvi = getNDVIHeatmap(image, self.cfg['HSI']['WV2PST'])
        mask = getMask(image, ndvi, self.cfg['HSI']['NDVI_THRESH'])
        #extract spectrum that is part of the leaf
        specs = image[mask]
        specs = torch.tensor(specs, dtype=torch.float32)
        # specs = specs.permute((1,0))
        # image = torch.tensor(image, dtype=torch.float32)
        # image = image.permute((2, 0, 1))
        # image = F.pad(image, (0, self.max_length-image.shape[2],0,self.max_width-image.shape[1]))
        # if self.transform:
        #     image = self.transform(image)
        return (specs, pst)

if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    dataset = SpecDataset(cfg, True)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample[0].size())