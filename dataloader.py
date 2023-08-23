import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import shutil
from option import parse_args

def process(root = "./dataset/Plane_data_750x800/train"):
    pathB = os.path.join(root, 'B')
    file_nameB = os.listdir(pathB)
    nameB = []
    grouped_files = {}
    
    for file_name in file_nameB :
        label = file_name.split("_")[0]  # Extract the label (C307, C118)
        direction = file_name.split("=")[1][3:5]  # Extract the direction (NW, SE)
        key = label + "_" + direction  # Compose a new key, e.g., "C307_NW"
        
        if key in grouped_files:
            grouped_files[key].append(file_name)
        else:
            grouped_files[key] = [file_name]
    for key, files in grouped_files.items():
        os.makedirs(os.path.join(root,f'{key}') , exist_ok = True)
        for target_name in files:
            for file_name in file_nameB:
                if target_name in file_name and file_name.endswith(".JPG"):
                    source_path = os.path.join(os.path.join(root,'B'), file_name)
                    target_path = os.path.join(os.path.join(root,f'{key}'), file_name)
                    shutil.copy(source_path, target_path)

class DamageIndexDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.pathA = os.path.join(opt.dataroot, 'train', 'A')
        self.pathB = os.path.join(opt.dataroot, 'train', 'B')
        self.df = pd.read_csv(os.path.join(opt.dataroot, "damageindex.csv"))
        self.continuous_column = opt.continuous_column
        self.discrete_column = opt.discrete_column
        self.con_label = self.df[self.continuous_column]
        self.con_attribute = self.con_label.columns.tolist()
        self.dis_label = self.df[self.discrete_column]
        self.preprocess()
        self.filenameB = os.listdir(self.pathB)
        self.transform= transform

    def preprocess(self):
        # One hot for discrete label
        self.dis_label = self.dis_label.applymap(str)
        self.dis_label = pd.get_dummies(self.dis_label)
        
        self.dis_attribute = self.dis_label.columns.to_list()

        self.dis_label = self.dis_label.values
        self.con_label = self.con_label.values
        self.label_nc = len(self.con_attribute) + len(self.dis_attribute)

    def __len__(self):
        """Return the number of images."""
        return len(self.filenameB)
        
    def __getitem__(self, index):
        filenameB = self.filenameB[index]
        DI = self.df.values[index][1]
        dis_label = self.dis_label[index]
        con_label = self.con_label[index]
        #label = filenameB.split("_")[0]  # Extract the label (C307, C118)
        #direction = filenameB.split("=")[1][3:5]  # Extract the direction (NW, SE)
        #filenameA = f"{label}_DI=0.0{direction}.JPG"
        filenameA = filenameB.replace(str(DI),"0.0")
        imagePathA = os.path.join(self.pathA, filenameA)
        imagePathB = os.path.join(self.pathB, filenameB)
        #print(filenameA, "    ", filenameB)
        #print(self.con_label, self.dis_label, self.con_attribute,self.dis_attribute) 
        imgA = self.transform(Image.open(imagePathA))
        imgB = self.transform(Image.open(imagePathB))
        
        # return {'imgA': imgA, 'imgB': imgB, 
        #         'self.con_attribute': self.con_attribute,
        #         'self.dis_attribute': self.dis_attribute,
        #         'label_nc': label_nc}
        return imgA, imgB, dis_label, con_label

if __name__ == "__main__":
    pass
    # input_size = 128
    # device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    # transform = transforms.Compose([transforms.Resize((input_size, input_size)), 
    #                                 transforms.ToTensor(), 
    #                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # (a,b,c,d) = DamageIndexDataset(transform=transform , opt=parse_args())
    # print(a,    b,    c,    d)
        #save_image(x, "x.png")
        #save_image(y, "y.png")
    #     import sys

    #     sys.exit()
