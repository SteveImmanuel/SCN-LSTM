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

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int):
        video_name = self.videos[idx]

        # get video data
        video_path = os.path.join(self.dataset_path, video_name)
        img_list = os.listdir(video_path)
        # max image sequence set to 50 frames, otherwise use sample rate
        step_size = max(len(img_list) // 50, int(1 / self.sample_rate))
        image_seq = []
        for i in range(0, len(img_list), step_size):
            image_path = os.path.join(video_path, img_list[i])
            image = read_image(image_path).type(torch.float32)

            for transform in self.transform:
                image = transform(image)
            image_seq.append(image.unsqueeze(0))

        image_seq_len = len(image_seq)
        assert image_seq_len > 0 and image_seq_len <= 80, video_name

        # output dim = (timestep, channel, height, width)
        # note that the video is not padded here, but in the model forward method
        # because I need to get the video sequence length during inferencing
        video_data = torch.cat(image_seq, 0)

        # get label
        annot_idx = random.randint(0, len(self.video_mapping[video_name]) - 1)
        annot_raw = self.video_mapping[video_name][annot_idx]
        # pad ending annotation with <BOS> until the length matches timestep
        annot_padded = annot_raw + [BOS_TAG] * (self.timestep - len(annot_raw) - (image_seq_len - 1))

        annotation = annotation_to_idx(annot_padded, self.word_to_idx)
        annotation = torch.FloatTensor(annotation)
        # pad beggining with zero
        pad_zero = torch.zeros(image_seq_len - 1)
        # output dim = (timestep)
        label_annotation = torch.cat([pad_zero, annotation], 0).long()
        annot_mask = torch.cat([
            torch.zeros(image_seq_len - 1),
            torch.ones(len(annot_raw)),
            torch.zeros(self.timestep - len(annot_raw) - (image_seq_len - 1))
        ], 0).long()

        return (video_data, (label_annotation, annot_mask))
