import os
import torch
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, Normalize
from s2vt.dataset import RawMSVDDataset
from s2vt.constant import *
from s2vt.utils import build_video_dict


def extract_features(annotations_file: str, root_path: str, output_dir: str, batch_size: int = 32) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vgg = models.vgg16(pretrained=True).to(DEVICE)
    # use output from fc7, remove the rest
    vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
    vgg.eval()

    all_videos = os.listdir(root_path)
    video_dict = build_video_dict(annotations_file)
    preprocess_funcs = [Normalize(VGG_MEAN, VGG_STD), RandomCrop(227)]

    for idx, video in enumerate(all_videos):
        print(f'Extracting video {idx+1}/{len(all_videos)}', end='\r')
        video_index = video_dict[video]
        output_video_path = os.path.join(output_dir, f'{video_index:04d}')
        os.makedirs(output_video_path, exist_ok=True)

        raw_dataset = RawMSVDDataset(os.path.join(root_path, video), video_dict, preprocess_funcs)
        data_loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False)

        for batch_idx, (X, _) in enumerate(data_loader):
            X = X.to(DEVICE)
            out = vgg(X)
            out_numpy = out.cpu().detach().numpy()

            for i in range(out_numpy.shape[0]):
                npy_path = os.path.join(output_video_path, f'frame{(batch_idx*batch_size)+i+1:04d}.npy')
                with open(npy_path, 'wb') as f:
                    np.save(f, out_numpy[i])

    print('\nFeature extraction complete')
