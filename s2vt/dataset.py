from cProfile import label
from dataclasses import dataclass
import os
import random
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import List
from s2vt.utils import *


class MSVDDataset(Dataset):
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
        return len(os.listdir(self.videos))

    def __getitem__(self, idx):
        video_name = self.videos[idx]

        # get label
        annot_idx = random.randint(0, len(self.video_mapping[video_name]) - 1)
        annot_raw = self.video_mapping[video_name][annot_idx]
        # pad annotation with <BOS> until the length matches timestep
        annot_raw += [BOS_TAG] * (self.timestep - len(annot_raw))
        annotation = annotation_to_idx(annot_raw, self.word_to_idx)
        label_annotation = torch.nn.functional.one_hot(
            torch.tensor(annotation),
            num_classes=self.vocab_size,
        )

        # get video data
        video_path = os.path.join(self.dataset_path, video_name)
        img_list = os.listdir(video_path)
        image_seq = []
        for i in range(0, len(img_list), int(1 / self.sample_rate)):
            image_path = os.path.join(video_path, img_list[i])
            image = read_image(image_path)

            for transform in self.transform:
                image = transform(image)
            image_seq.append(image.unsqueeze(0))

        assert len(image_seq) > 0

        image_dim = image_seq[0].shape
        pad_zero = torch.zeros((self.timestep - len(image_seq), *image_dim[1:]))
        image_seq.append(pad_zero)
        video_data = torch.cat(image_seq, 0)

        return (video_data, label_annotation)
