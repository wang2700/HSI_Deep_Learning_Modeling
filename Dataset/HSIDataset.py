import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from files import Files
from utils import caliColor

class HSIDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=250):
        self.root_dir = root_dir
        self.transform = transform
        mfiles = Files(self.root_dir, ['*heatmap.npy'])
        self.rawNameList = mfiles.filesNoExt
        self.rawNameList = [n[:-8] for n in self.rawNameList]
        self.max_length = max_length

    def __len__(self): 
        return len(self.rawNameList)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '.npy')
        whiteRef = np.load(self.root_dir + r'/' + self.rawNameList[idx] + '_whRef.npy')
        image = caliColor(image, whiteRef)


        image = torch.tensor(image)
        image = image.permute((2, 0, 1))
        if self.transform:
            image = self.transform(image)
        return image
    
    



if __name__ == "__main__":
    root_dir = r'/media/jerrynas/Research/LeafSpec/SoyBean Device Data/Sumitomo 2020/Raw Data'
    transform_list = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)#,
        # transforms.RandomRotation((-180, 180), expand=True)
    )
    
    dataset = HSIDataset(root_dir=root_dir, transform=transform_list)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample.size())