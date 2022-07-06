import os
import math
import random
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

        return (tensor_frames, torch.empty(0))  # C, 16, H, W


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

        occurence = list(map(lambda x: x[1], self.tag_dict.keys()))
        self.weight_mask = torch.FloatTensor(occurence)**.5

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

        tags = {key[0]: value for key, value in self.tag_dict.items()}

        label = torch.zeros(len(tags))
        unique_words = set()
        annotations = self.video_caption_mapping[video_name]
        for annotation in annotations:
            for token in annotation:
                unique_words.add(token)
        for word in unique_words:
            if word in tags:
                label[tags[word]] = 1.0

        return (cnn_features, label, self.weight_mask)


class CompiledMSVD(Dataset):
    """
    Consist of extracted cnn features, extracted semantic features,
    and captions
    """
    def __init__(
        self,
        annotation_file: str,
        cnn_features_path: str,
        semantic_features_path: str,
        beta: float = 0,
        timestep: int = 80,
        max_len: int = -1,
    ) -> None:
        super().__init__()
        assert os.path.exists(cnn_features_path)
        assert os.path.exists(semantic_features_path)
        assert os.path.exists(annotation_file)

        self.cnn_features_path = cnn_features_path
        self.semantic_features_path = semantic_features_path
        self.word_to_idx, self.idx_to_word, self.video_caption_mapping = build_vocab(annotation_file)
        self.video_dict = build_video_dict(annotation_file, reverse_key=True)
        self.all_cnn_features = sorted(os.listdir(cnn_features_path))
        self.all_semantic_features = sorted(os.listdir(semantic_features_path))
        self.videos = list(map(lambda x: x[5:9], self.all_cnn_features))
        self.vocab_size = len(self.word_to_idx)
        self.timestep = timestep
        self.max_len = max_len

        self.timestep_weight = 1 / (torch.arange(0, timestep)**beta)

    def __len__(self):
        if self.max_len == -1:
            return len(self.videos)
        else:
            return self.max_len

    def __getitem__(self, index):
        video_idx = int(self.videos[index])
        video_name = self.video_dict[video_idx]

        cnn_features_path = os.path.join(self.cnn_features_path, self.all_cnn_features[index])
        cnn_features = np.load(cnn_features_path)
        cnn_features = torch.FloatTensor(cnn_features)

        semantic_features_path = os.path.join(self.semantic_features_path, self.all_semantic_features[index])
        semantic_features = np.load(semantic_features_path)
        semantic_features = torch.FloatTensor(semantic_features)

        # get label
        annot_idx = random.randint(0, len(self.video_caption_mapping[video_name]) - 1)
        annot_raw = self.video_caption_mapping[video_name][annot_idx]  # already contains BOS and EOS
        # pad ending annotation with <EOS> until the length matches timestep
        annot_padded = annot_raw + [EOS_TAG] * (self.timestep - len(annot_raw))

        annotation = annotation_to_idx(annot_padded, self.word_to_idx)
        label_annotation = torch.LongTensor(annotation)
        # output dim = (timestep)

        assert self.timestep - len(
            annot_raw) >= 0, f'Annotation too long for video {video_name}, len={len(annot_raw)} words'

        annot_mask = torch.cat(
            [
                torch.zeros(1),  # BOS tag
                torch.ones(len(annot_raw) - 1),  # annotation + EOS tag
                torch.zeros(self.timestep - len(annot_raw)),
            ],
            0,
        ).long()
        weighted_mask = annot_mask * self.timestep_weight

        return (cnn_features, semantic_features, label_annotation, weighted_mask)


class ExtractedMSVD(Dataset):
    def __init__(
        self,
        annotation_file: str,
        cnn_features_path: str,
        semantic_features_path: str,
        type: str,  # train or test
        beta: float = 0,
        timestep: int = 80,
        max_len: int = -1,
        max_sentence_len: int = 25,
        include_validation_train: bool = True,
    ) -> None:
        super().__init__()
        assert os.path.exists(cnn_features_path)
        assert os.path.exists(semantic_features_path)
        assert os.path.exists(annotation_file)
        assert timestep >= max_sentence_len

        self.type = type
        self.include_validation_train = include_validation_train
        self.word_to_idx, self.idx_to_word, self.video_caption_mapping = build_vocab(annotation_file)
        self.video_dict = build_video_dict(annotation_file, reverse_key=True)
        self.vocab_size = len(self.word_to_idx)
        self.timestep = timestep
        self.max_len = max_len
        self.max_sentence_len = max_sentence_len
        self.timestep_weight = 1 / (torch.arange(0, timestep)**beta)
        self.all_cnn_features = np.load(cnn_features_path)
        self.all_semantic_features = np.load(semantic_features_path)

    def __len__(self):
        if self.type == 'train':
            if self.include_validation_train:
                return 1300
            else:
                return 1200
        else:
            if self.max_len == -1:
                return 670
            else:
                return self.max_len

    def __getitem__(self, index):
        if self.type != 'train':
            index += 1300
        video_name = self.video_dict[index]

        # get label
        annot_idx = random.randint(0, len(self.video_caption_mapping[video_name]) - 1)
        annot_raw = self.video_caption_mapping[video_name][annot_idx]  # already contains BOS and EOS
        while len(annot_raw) > self.max_sentence_len:
            annot_idx = random.randint(0, len(self.video_caption_mapping[video_name]) - 1)
            annot_raw = self.video_caption_mapping[video_name][annot_idx]  # already contains BOS and EOS

        # pad ending annotation with <EOS> until the length matches timestep
        annot_padded = annot_raw + [EOS_TAG] * (self.timestep - len(annot_raw))

        annotation = annotation_to_idx(annot_padded, self.word_to_idx)
        label_annotation = torch.LongTensor(annotation)  # output dim = (timestep)

        annot_mask = torch.cat(
            [
                torch.zeros(1),  # BOS tag
                torch.ones(len(annot_raw) - 1),  # annotation + EOS tag
                torch.zeros(self.timestep - len(annot_raw)),
            ],
            0,
        ).long()
        weighted_mask = annot_mask * self.timestep_weight

        cnn_features = torch.FloatTensor(self.all_cnn_features[index])
        semantic_features = torch.FloatTensor(self.all_semantic_features[index])

        return (
            cnn_features,
            semantic_features,
            label_annotation,
            weighted_mask,
        )
