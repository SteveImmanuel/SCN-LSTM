import os
import math
import torch
import cv2
import numpy as np
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


class CNNExtractedMSVD(Dataset):
    def __init__(
        self,
        annotation_file: str,
        root_path: str,
        num_tags: int,
        cnn_2d_model: str = 'regnetx32',
        cnn_3d_model: str = 'shufflenetv2',
    ) -> None:
        super().__init__()
        self.cnn_2d_model = cnn_2d_model
        self.cnn_3d_model = cnn_3d_model
        self.root_path = root_path
        self.video_dict = build_video_dict(annotation_file, reverse_key=True)
        self.word_to_idx, self.idx_to_word, self.video_caption_mapping = build_vocab(annotation_file)
        self.tag_dict = build_tags(annotation_file, num_tags=num_tags, reverse_key=True)
        self.videos = list(set(map(lambda x: x[5:9], os.listdir(self.root_path))))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_idx = self.videos[index]
        video_name = self.video_dict[int(video_idx)]

        cnn2d_npy_path = os.path.join(self.root_path, f'video{video_idx}_cnn_2d_{self.cnn_2d_model}.npy')
        cnn3d_npy_path = os.path.join(self.root_path, f'video{video_idx}_cnn_3d_{self.cnn_3d_model}.npy')
        cnn2d_features = torch.FloatTensor(np.load(cnn2d_npy_path))
        cnn3d_features = torch.FloatTensor(np.load(cnn3d_npy_path))
        cnn_features = torch.cat((cnn3d_features, cnn2d_features), dim=0)

        label = torch.zeros(len(self.tag_dict))
        unique_words = set()
        annotations = self.video_caption_mapping[video_name]
        for annotation in annotations:
            for token in annotation:
                unique_words.add(token)
        for word in unique_words:
            if word in self.tag_dict:
                label[self.tag_dict[word]] = 1.0

        return (cnn_features, label)
