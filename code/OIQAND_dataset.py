import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class JUFE_10K_Dataset(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        idx_list = [str(i) for i in range(8)]
        column_names = idx_list + ['mos']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(8):
            p1, p2, p3 = self.X.iloc[index, i].split("/")
            path = os.path.join("/home/d310/10t/rjl/dataset", p1, p2, p3)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)

        # # 8vps乱序
        vs1 = img_list
        vs1 = torch.cat(vs1)
        imgs = vs1

        mos = torch.FloatTensor(np.array(self.mos[index]))
        
        sample = {
            'd_img_org': imgs,
            'score': mos,
        }

        return sample
    

class JUFE_10K_Dataset_v20(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        idx_list = [str(i) for i in range(20)]
        column_names = idx_list + ['mos']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(20):
            p1, p2, p3 = self.X.iloc[index, i].split("/")
            path = os.path.join("/home/d310/10t/rjl/dataset", p1, p2, p3) #/home/d310/10t/rjl/dataset
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)

        # # 8vps乱序
        vs1 = img_list
        vs1 = torch.cat(vs1)
        imgs = vs1

        mos = torch.FloatTensor(np.array(self.mos[index]))
        
        sample = {
            'd_img_org': imgs,
            'score': mos,
        }

        return sample

if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = JUFE_10K_Dataset(info_csv_path="/home/d310/10t/rjl/TMM_OIQA/file/VGCN/test_VGCN_viewport8.csv", transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=8,
        shuffle=False,
    )
    print(len(test_dataset))
    for data in test_loader:
        imgs = data['d_img_org']
        mos = data['score']
        print(imgs.shape)
        print(mos.shape)
        print(mos)
        break



    
        