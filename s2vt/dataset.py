import os
import random
import numpy as np
import math
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import List
from s2vt.utils import *


class RawMSVDDataset(Dataset):
    """
    Raw image only, for feature extraction
    """
    def __init__(
        self,
        video_path: str,
        video_dict: Dict,
        transform: List = None,
    ):
        assert os.path.exists(video_path)
        self.video_path = video_path
        self.images = sorted(os.listdir(video_path))
        self.transform = transform
        self.video_dict = video_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.video_path, self.images[idx])
        image = read_image(image_path).type(torch.float32) / 255.0
        for transform in self.transform:
            image = transform(image)

        return (image, torch.empty(0))


class PreprocessedMSVDDataset(Dataset):
    """
    Preprocessed raw image into extracted features and caption.
    Image sequence and caption will both have the timestep sequences.
    Image will be padded on the right, caption will be padded on the left
    """
    def __init__(
        self,
        dataset_path: str,
        annotation_file: str,
        timestep: int = 80,
    ):
        assert os.path.exists(dataset_path)
        assert os.path.exists(annotation_file)

        self.dataset_path = dataset_path
        self.videos = sorted(os.listdir(dataset_path))
        self.word_to_idx, self.idx_to_word, self.video_caption_mapping = build_vocab(annotation_file)
        self.video_dict = build_video_dict(annotation_file, reverse_key=True)
        self.vocab_size = len(self.word_to_idx)
        self.timestep = timestep

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int):
        video_idx = self.videos[idx]
        video_name = self.video_dict[int(video_idx)]

        # get video data
        video_path = os.path.join(self.dataset_path, video_idx)
        frame_list = sorted(os.listdir(video_path))
        # set frame sequence len to set to timestep, pad with zero until the len is timestep if neccessary
        step_size = max(math.ceil(len(frame_list) / self.timestep), 1)

        frame_seq = []
        for i in range(0, len(frame_list), step_size):
            npy_path = os.path.join(video_path, frame_list[i])
            frame_features = np.load(npy_path)
            frame_seq.append(frame_features)

        # pad with zero for the remaining timestep
        pad_zero = np.zeros((self.timestep - len(frame_seq), *frame_seq[0].shape))
        frame_seq = np.concatenate((pad_zero, frame_seq), axis=0)
        # output dim = (timestep, feature_dim)
        video_data = torch.FloatTensor(frame_seq)

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
            0).long()

        return (video_data, (label_annotation, annot_mask))


class EndToEndMSVDDataset(Dataset):
    """
    Complete from raw image and caption. DEPRECATED DO NOT USE
    """
    def __init__(
        self,
        dataset_path: str,
        annotation_file: str,
        transform: List = None,
        timestep: int = 80,
        sample_rate: float = 0.1,
    ):
        assert os.path.exists(dataset_path)
        assert os.path.exists(annotation_file)

        self.dataset_path = dataset_path
        self.videos = os.listdir(dataset_path)
        self.transform = transform
        self.word_to_idx, self.idx_to_word, self.video_mapping = build_vocab(annotation_file)
        self.vocab_size = len(self.word_to_idx)
        self.timestep = timestep
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int):
        video_name = self.videos[idx]

        # get video data
        video_path = os.path.join(self.dataset_path, video_name)
        img_list = os.listdir(video_path)
        # max image sequence set to 45 frames, otherwise use sample rate
        step_size = max(len(img_list) // 45, int(1 / self.sample_rate))
        image_seq = []
        for i in range(0, len(img_list), step_size):
            image_path = os.path.join(video_path, img_list[i])
            image = read_image(image_path).type(torch.float32) / 255.0

            for transform in self.transform:
                image = transform(image)
            image_seq.append(image.unsqueeze(0))

        image_seq_len = len(image_seq)
        assert image_seq_len > 0 and image_seq_len <= 50, f'Video too long {video_name}, len={image_seq_len} frames'

        image_dim = image_seq[0].shape
        # pad with zero for the remaining timestep
        pad_zero = torch.zeros(self.timestep - image_seq_len, *image_dim[1:])
        image_seq.append(pad_zero)
        video_data = torch.cat(image_seq, 0)

        # get label
        annot_idx = random.randint(0, len(self.video_mapping[video_name]) - 1)
        annot_raw = self.video_mapping[video_name][annot_idx]
        # pad ending annotation with <BOS> until the length matches timestep
        annot_padded = annot_raw + [EOS_TAG] * (self.timestep - len(annot_raw) - (image_seq_len - 1))

        annotation = annotation_to_idx(annot_padded, self.word_to_idx)
        annotation = torch.FloatTensor(annotation)
        # pad beggining with zero
        pad_zero = torch.zeros(image_seq_len - 1)
        # output dim = (timestep)
        label_annotation = torch.cat([pad_zero, annotation], 0).long()

        assert self.timestep - len(annot_raw) - (
            image_seq_len - 1) >= 0, f'Annotation too long for video {video_name}, len={len(annot_raw)} words'

        annot_mask = torch.cat([
            torch.zeros(image_seq_len - 1),
            torch.ones(len(annot_raw)),
            torch.zeros(self.timestep - len(annot_raw) - (image_seq_len - 1))
        ], 0).long()

        return ((video_data, image_seq_len), (label_annotation, annot_mask))
