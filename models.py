import torch
import torch.nn as nn
from utils import count_model_params

import warnings as wrn
wrn.filterwarnings('ignore')

class RiceLeafNet(nn.Module):
    def __init__(self):
        super(RiceLeafNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=72, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=72, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=56, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=56, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=2, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(num_features=64)
        self.conv2_bn = nn.BatchNorm2d(num_features=72)
        self.conv3_bn = nn.BatchNorm2d(num_features=64)
        self.conv4_bn = nn.BatchNorm2d(num_features=56)
        self.conv5_bn = nn.BatchNorm2d(num_features=64)
        self.conv6_bn = nn.BatchNorm2d(num_features=80)
        self.fc1 = nn.Linear(in_features=80, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=5)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.2)

    def forward(self, img):
        img = self.maxpool(self.relu(self.dropout(self.conv1(img))))
        img = self.maxpool(self.relu(self.dropout(self.conv2(img))))
        img = self.maxpool(self.relu(self.dropout(self.conv3(img))))
        img = self.maxpool(self.relu(self.dropout(self.conv4(img))))
        img = self.maxpool(self.relu(self.dropout(self.conv5(img))))
        img = self.relu(self.dropout(self.conv6(img)))
        img = img.reshape(img.shape[0], -1)
        # img = torch.flatten(img)
        img = self.relu(self.dropout(self.fc1(img)))
        img = self.fc2(img)
        return img

    # def forward(self, img):
    #     img = self.maxpool(self.relu(self.conv1_bn(self.dropout(self.conv1(img)))))
    #     img = self.maxpool(self.relu(self.conv2_bn(self.dropout(self.conv2(img)))))
    #     img = self.maxpool(self.relu(self.conv3_bn(self.dropout(self.conv3(img)))))
    #     img = self.maxpool(self.relu(self.conv4_bn(self.dropout(self.conv4(img)))))
    #     img = self.maxpool(self.relu(self.conv5_bn(self.dropout(self.conv5(img)))))
    #     img = self.relu(self.conv6_bn(self.dropout(self.conv6(img))))
    #     img = img.reshape(img.shape[0], -1)
    #     # img = torch.flatten(img)
    #     img = self.relu(self.dropout(self.fc1(img)))
    #     img = self.fc2(img)
    #     return img

def test():
    rand_batch = torch.randn(32, 3, 128, 128)
    model = RiceLeafNet()
    model_outputs = model(rand_batch)
    print(model_outputs.shape)
    print(f"Total Parameters: {count_model_params(model)}")

if __name__ == "__main__":
    test()
