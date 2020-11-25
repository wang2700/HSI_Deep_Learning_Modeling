import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from .files import Files
from utils.HSI_Analysis import caliColor
import csv

class HSIDataset(Dataset):
    def __init__(self, cfg, root_dir, max_length, max_width, transform=None, train=True):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.max_width = max_width
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
        self.max_length = max_length

    def __len__(self): 
        return len(self.rawNameList)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        plot_name = self.rawNameList[idx][14:18]
        P_gt = self.cfg['DATASET']['CLASSES'][plot_name]

        image = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '.npy')
        whiteRef = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '_whRef.npy')
        pst = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '_imgPst.npy')
        image = caliColor(image, whiteRef)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute((2, 0, 1))
        image = F.pad(image, (0, self.max_length-image.shape[2],0,self.max_width-image.shape[1]))
        if self.transform:
            image = self.transform(image)
        return (image, pst, P_gt)
    
    



if __name__ == "__main__":
    root_dir = r'/media/jerrynas/Research/LeafSpec/SoyBean Device Data/Sumitomo 2020/Raw Data'
    transform_list = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((-20, 20), expand=False)
    )
    
    dataset = HSIDataset(root_dir=root_dir, transform=transform_list, max_length=204)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample.size())