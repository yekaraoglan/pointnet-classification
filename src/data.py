from path import Path
import os
import numpy as np
import math
import random
random.seed(42)
import torch

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from util import read_off_file, visualize_rotate, pcshow, MeshToPointCloud
from preprocessing import Normalize, RandomRotationInZ, RandomNoise, ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

path = Path('../data/ModelNet10')
folders = [folder for folder in sorted(os.listdir(path)) if os.path.isdir(path/folder)]
classes = {folder: i for i, folder in enumerate(folders)}

class PointCloudData(Dataset):
    def __init__(self, path, transforms=None, valid=False, folder='train'):
        self.files = []
        self.transforms = transforms
        self.valid = valid
        self.folder = folder

        for category in os.listdir(path):
            for file in os.listdir(path/category/folder):
                sample = {'path': path/category/folder/file, 'category': category}
                self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        with open(file, 'r') as f:
            verts, faces = read_off_file(f)
            if self.transforms:
                pc = self.transforms((verts, faces))
            return pc

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['path']
        category = self.files[idx]['category']
        category = torch.tensor(classes[category])
        pc = self.__preproc__(pcd_path)
        return pc,  category

def train_transforms():
    return transforms.Compose([
        MeshToPointCloud(1024),
        Normalize(),
        RandomRotationInZ(),
        # RandomNoise(),
        ToTensor()
    ])

def test_transforms():
    return transforms.Compose([
        MeshToPointCloud(1024),
        Normalize(),
        ToTensor()
    ])

train_set = PointCloudData(path, transforms=train_transforms(), folder='train')
valid_set = PointCloudData(path, transforms=test_transforms(), folder='test')
test_set = PointCloudData(path, transforms=test_transforms(), folder='test')

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)
