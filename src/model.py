import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import math

class TransformNet(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        assert isinstance(k, int)
        self.k = k
        self.conv1 = nn.Conv1d(self.k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k * self.k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        max_x = torch.max(x, 1, keepdim=True)[0]
        x = max_x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k, requires_grad=True).repeat(x.size(0), 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x.view(-1, self.k, self.k) + identity

        input = input.transpose(2, 1)
        return torch.matmul(input, x).transpose(2, 1)

class PointNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        assert isinstance(num_classes, int)
        self.num_classes = num_classes
        self.input_transform = TransformNet(3)
        self.feature_transform = TransformNet(64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)

    def forward(self, x):
        m3 = self.input_transform(x)
        x = self.relu(self.bn1(self.conv1(m3)))
        x = self.relu(self.bn2(self.conv2(x)))
        m64 = self.feature_transform(x)
        x = self.relu(self.bn3(self.conv3(m64)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)
        x = x.view(-1, 1024)
        x = self.relu(self.bn6(self.fc1(x)))
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), m3, m64

def pointnet_loss(output, label, m3, m64, alpha=0.0001):
    criterion = nn.NLLLoss()
    batch_size = output.size(0)
    id3 = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1)
    id64 = torch.eye(64, requires_grad=True).repeat(batch_size, 1, 1)
    if output.is_cuda:
        id3 = id3.cuda()
        id64 = id64.cuda()
    diff3 = id3 - torch.bmm(m3, m3.transpose(2, 1))
    diff64 = id64 - torch.bmm(m64, m64.transpose(2, 1))
    return criterion(output, label) + alpha * (torch.norm(diff3) + torch.norm(diff64)) / float(batch_size)