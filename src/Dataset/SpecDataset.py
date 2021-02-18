import time
import sys
sys.path.insert(0, '/home/jerry/Documents/Research/HSI_Deep_Learning_Modeling/')
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from .utils.files import Files
from .utils.HSI_Analysis import caliColor, getMask, getNDVIHeatmap
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
        #find index of the spectrum that got extracted.
        location = np.where(mask)
        #extract spectrum that is part of the leaf
        specs = image[mask]
        specs = torch.tensor(specs, dtype=torch.float32)
        
        # t0 = time.time()
        # reconstruct = torch.tensor(np.zeros_like(image), dtype=torch.float32)
        # k = 0
        # for k in range(len(location[0])):
        #     i = location[0][k]
        #     j = location[1][k]
        #     reconstruct[i][j] = specs[k]
        #     k += 1
        # t1 = time.time()
        # print(t1-t0)
        

        # specs = specs.permute((1,0))
        # image = torch.tensor(image, dtype=torch.float32)
        # image = image.permute((2, 0, 1))
        # image = F.pad(image, (0, self.max_length-image.shape[2],0,self.max_width-image.shape[1]))
        # if self.transform:
        #     image = self.transform(image)
        return (specs, pst, location, image.shape)

if __name__ == "__main__":
    cfg = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
    dataset = SpecDataset(cfg, True)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample[0].size())