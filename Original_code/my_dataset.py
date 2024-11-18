import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms as transforms
from torch.utils.data import Dataset

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class O10K_Dataset(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        # idx_list = [str(i) for i in range(15)]
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
            path = os.path.join("/mnt/10T/rjl/dataset", p1, p2, p3)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)

        img1 = img_list[0]
        img3 = img_list[1]
        img2 = img_list[2]
        img4 = img_list[3]
        img7 = img_list[4]
        img8 = img_list[5]
        img5 = img_list[6]
        img6 = img_list[7]
        

        # # 8vps乱序
        vs1 = img_list
        vs2 = [img_list[0],img_list[1],img_list[2],img_list[3],img_list[3],img_list[2],img_list[1],img_list[0]]
        vs3 = [img_list[4],img_list[5],img_list[6],img_list[7],img_list[7],img_list[6],img_list[5],img_list[4]]

        vs1 = torch.cat(vs1).unsqueeze(0)
        vs2 = torch.cat(vs2).unsqueeze(0)
        vs3 = torch.cat(vs3).unsqueeze(0)

        # # 8vps有序
        # imgs1 = [img1, img2, img3, img4, img5, img6, img7, img8]
        # imgs2 = [img1, img2, img3, img4, img4, img3, img2, img1]
        # imgs3 = [img5, img6, img7, img8, img8, img7, img6, img5]

        

        # imgs1 = torch.cat(imgs1).unsqueeze(0)
        # imgs2 = torch.cat(imgs2).unsqueeze(0)
        # imgs3 = torch.cat(imgs3).unsqueeze(0)

        # imgs = torch.cat((vs1, vs2, vs3), dim=0)
        # imgs = torch.cat(vs1, dim=0)
        
        #三个8vps 乱序
        # imgs = torch.cat((vs1, vs2, vs3), dim=0)
        #一个8vps 乱序
        imgs = vs1

        #一个8vps 有序
        # imgs = imgs1

        #三个8vps 有序
        # imgs = torch.cat((imgs1, imgs2, imgs3), dim=0)

        mos = torch.FloatTensor(np.array(self.mos[index]))
        
        sample = {
            'd_img_org': imgs,
            'score': mos,
        }

        return sample
    
# class O10K_Dataset(Dataset):
#     def __init__(self, info_csv_path, transform=None):
#         super().__init__()
#         self.transform = transform
#         # idx_list = [str(i) for i in range(15)]
#         idx_list = [str(i) for i in range(8)]
#         column_names = idx_list + ['n'] + ['mos']
#         self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
#         self.X = self.df[idx_list]
#         self.mos = self.df['mos']
        
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, index):
#         img_list = []
#         for i in range(8):
#             p1, p2 = self.X.iloc[index, i].split("/")
#             path = os.path.join("/home/test/10t/tzw/methods/dataset/OIQ-10K/viewports_8", p1, p2)
#             img = Image.open(path)
        
#             if self.transform:
#                 img = self.transform(img)
#             img = img.float().unsqueeze(0)
#             img_list.append(img)

#         img1 = img_list[0]
#         img3 = img_list[1]
#         img2 = img_list[2]
#         img4 = img_list[3]
#         img7 = img_list[4]
#         img8 = img_list[5]
#         img5 = img_list[6]
#         img6 = img_list[7]
        

#         # # 8vps乱序
#         vs1 = img_list
#         vs2 = [img_list[0],img_list[1],img_list[2],img_list[3],img_list[3],img_list[2],img_list[1],img_list[0]]
#         vs3 = [img_list[4],img_list[5],img_list[6],img_list[7],img_list[7],img_list[6],img_list[5],img_list[4]]

#         vs1 = torch.cat(vs1).unsqueeze(0)
#         vs2 = torch.cat(vs2).unsqueeze(0)
#         vs3 = torch.cat(vs3).unsqueeze(0)

#         # # 8vps有序
#         # imgs1 = [img1, img2, img3, img4, img5, img6, img7, img8]
#         # imgs2 = [img1, img2, img3, img4, img4, img3, img2, img1]
#         # imgs3 = [img5, img6, img7, img8, img8, img7, img6, img5]

        

#         # imgs1 = torch.cat(imgs1).unsqueeze(0)
#         # imgs2 = torch.cat(imgs2).unsqueeze(0)
#         # imgs3 = torch.cat(imgs3).unsqueeze(0)

#         # imgs = torch.cat((vs1, vs2, vs3), dim=0)
#         # imgs = torch.cat(vs1, dim=0)
        
#         #三个8vps 乱序
#         # imgs = torch.cat((vs1, vs2, vs3), dim=0)
#         #一个8vps 乱序
#         imgs = vs1

#         #一个8vps 有序
#         # imgs = imgs1

#         #三个8vps 有序
#         # imgs = torch.cat((imgs1, imgs2, imgs3), dim=0)

#         mos = torch.FloatTensor(np.array(self.mos[index]))
        
#         sample = {
#             'd_img_org': imgs,
#             'score': mos,
#         }

#         return sample
    

if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = O10K_Dataset(info_csv_path="/home/test/10t/tzw/methods/dataset/OIQ-10K/OIQ-10K_train_info.csv", transform=test_transform)
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



    
        