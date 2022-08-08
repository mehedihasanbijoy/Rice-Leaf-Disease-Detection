import torch
import torchvision
import glob
import pandas as pd
import cv2

class RiceDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.df.iloc[idx, 2:].values.astype(float))
        return img, label



if __name__ == "__main__":
    root_dir = './Dataset/'
    rice_dataset = RiceDataset(root_dir, None)
    print(rice_dataset.__df__())