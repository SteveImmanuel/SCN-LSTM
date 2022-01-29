import os
import random
from matplotlib import transforms
import numpy as np
import math
import torch
import cv2
from torchvision.transforms import Compose
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import List
from utils import *


class RawMSVDDataset(Dataset):
    """
    Raw image only
    """
    def __init__(self, video_path: str, transform: List = None):
        assert os.path.exists(video_path)
        self.video_path = video_path
        self.images = sorted(os.listdir(video_path))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.video_path, self.images[idx])
        image = read_image(image_path).type(torch.float32) / 255.0
        for transform in self.transform:
            image = transform(image)

        return (image, torch.empty(0))


class SequenceImageMSVDDataset(Dataset):
    """
    Sequence of 16 images in BGR format in range [0,255] to be fed to 3D CNN models
    """
    def __init__(self, video_path: str, transform: List = None) -> None:
        super().__init__()
        self.video_path = video_path
        self.frames = sorted(os.listdir(video_path))
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        total_frames = len(self.frames)
        step_size = max(math.floor(total_frames / 16), 1)  # assume there is no video that has less than 16 frames

        frames = []
        for i in range(0, total_frames, step_size):
            if len(frames) == 16:
                break

            frame_path = os.path.join(self.video_path, self.frames[i])
            frame = cv2.imread(frame_path)
            frame = frame.transpose((2, 0, 1))  # H,W,C to C,H,W

            frame = torch.FloatTensor(frame)

            for transform in self.transform:
                frame = transform(frame)
            frame = frame.unsqueeze(1)
            frames.append(torch.FloatTensor(frame))

        tensor_frames = torch.cat(frames, dim=1)

        return (tensor_frames, torch.empty(0))  # 16, C, H, W