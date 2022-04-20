from torchvision import transforms
from torch.utils.data import Dataset

import torch
from PIL import Image
import os

class Dataset(Dataset):
    def __init__(self, root, transform, mode='train'):
        super(Dataset, self).__init__()

        self.root = root
        self.transform = transform
        self.mode = mode

        data_dir = os.path.join(root, mode)
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.mode, self.file_list[index])
        img = Image.open(img_path)
        img_out = self.transform(img)

        return img_out


def data_loader(root, batch_size=20, shuffle=True, img_size=32, mode='train'):    
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
    
    dset = Dataset(root, transform, mode=mode)
    
    if batch_size == 'all':
        batch_size = len(dset)
        
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=0,
                                          drop_last=True)
    dlen = len(dset)
    
    return dloader, dlen