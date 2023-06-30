import numpy as np
import random
import torch

class Normalize(object):
    def __call__(self, sample):
        assert isinstance(sample, np.ndarray)
        assert len(sample.shape) == 2

        normalized_pc = sample - np.mean(sample, axis=0)
        normalized_pc = normalized_pc / np.max(np.linalg.norm(normalized_pc, axis=1))

        return normalized_pc

class RandomRotationInZ(object):
    def __call__(self, sample):
        assert isinstance(sample, np.ndarray)
        assert len(sample.shape) == 2
        rand_radian = random.random() * 2 * np.pi
        self.rotation_matrix = np.array([[np.cos(rand_radian), -np.sin(rand_radian), 0],
                                            [np.sin(rand_radian), np.cos(rand_radian), 0],
                                            [0, 0, 1]])
        return np.matmul(sample, self.rotation_matrix)

class RandomNoise(object):
    def __call__(self, sample):
        assert isinstance(sample, np.ndarray)
        assert len(sample.shape) == 2

        noise = np.random.normal(0, 0.02, sample.shape)
        return sample + noise

class ToTensor(object):
    def __call__(self, sample):
        assert isinstance(sample, np.ndarray)
        assert len(sample.shape) == 2

        return torch.from_numpy(sample).float()
