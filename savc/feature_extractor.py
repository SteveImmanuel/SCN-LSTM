import os
import torch
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, Normalize, Resize
from savc.dataset import RawMSVDDataset, SequenceImageMSVDDataset
from savc.models.shufflenet import get_model as ShuffleNet
from savc.models.shufflenetv2 import get_model as ShuffleNetV2
from constant import *
from utils import build_video_dict


def extract_features_2d_cnn(annotations_file: str, root_path: str, output_dir: str, batch_size: int = 32) -> None:
    os.makedirs(output_dir, exist_ok=True)
    regnet_x_32gf = models.regnet_x_32gf(pretrained=True).to(DEVICE)
    # remove last layer
    regnet_x_32gf.fc = torch.nn.Identity()
    regnet_x_32gf.eval()

    all_videos = os.listdir(root_path)
    video_dict = build_video_dict(annotations_file)
    preprocess_funcs = [Normalize(IMAGE_MEAN, IMAGE_STD), RandomCrop(227)]

    for idx, video in enumerate(all_videos):
        print(f'Extracting video {idx+1}/{len(all_videos)}', end='\r')
        video_index = video_dict[video]

        raw_dataset = RawMSVDDataset(os.path.join(root_path, video), preprocess_funcs)
        data_loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False)

        res = None
        total_frames = 0
        for _, (X, _) in enumerate(data_loader):
            X = X.to(DEVICE)
            out = regnet_x_32gf(X)
            total_frames += out.shape[0]

            out = torch.sum(out, dim=0)
            out_numpy = out.cpu().detach().numpy()

            if res is None:
                res = out_numpy
            else:
                res += out_numpy

        res /= total_frames
        npy_path = os.path.join(output_dir, f'video{video_index:04d}_cnn_2d_regnetx32.npy')
        with open(npy_path, 'wb') as f:
            np.save(f, res)

    print('\nFeature extraction complete')


def extract_features_3d_cnn(
    annotations_file: str,
    root_path: str,
    output_dir: str,
    model_name: str = 'shufflenetv2',
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if model_name == 'shufflenetv2':
        model = ShuffleNetV2(
            './checkpoints/shufflenet/kinetics_shufflenetv2_2.0x_RGB_16_best.pth',
            width_mult=2.,
        )
    elif model_name == 'shufflenet':
        model = ShuffleNet(
            './checkpoints/shufflenet/kinetics_shufflenet_2.0x_G3_RGB_16_best.pth',
            width_mult=2.,
        )
    else:
        assert False, f'Unsupported model {model_name}'
    model.eval()

    all_videos = os.listdir(root_path)
    video_dict = build_video_dict(annotations_file)
    preprocess_funcs = [Normalize(KINETICS_MEAN, KINETICS_STD), Resize((112, 112))]

    for idx, video in enumerate(all_videos):
        print(f'Extracting video {idx+1}/{len(all_videos)}', end='\r')
        video_index = video_dict[video]

        raw_dataset = SequenceImageMSVDDataset(os.path.join(root_path, video), preprocess_funcs)
        data_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)

        for _, (X, _) in enumerate(data_loader):
            X = X.to(DEVICE)
            out = model(X)
            out_numpy = out[0].cpu().detach().numpy()

        npy_path = os.path.join(output_dir, f'video{video_index:04d}_cnn_3d_{model_name}.npy')
        with open(npy_path, 'wb') as f:
            np.save(f, out_numpy)

    print('\nFeature extraction complete')


def parse_features_from_txt(feature_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    video_set = set()
    with open(feature_file, 'r') as f:
        line = f.readline()
        while line:
            tokens = line.split(',')
            info = tokens[0].split('_')
            frame_idx = int(info[-1])
            video_idx = int(info[0][3:])
            video_name = f'{video_idx-1:04}'

            video_path = os.path.join(output_dir, video_name)
            if video_name not in video_set:
                video_set.add(video_name)
                os.makedirs(video_path, exist_ok=True)

            print(' ' * 80, end='\r')
            print(f'Parsing {video_name}, frame {frame_idx}', end='\r')

            features = tokens[1:]
            features = list(map(lambda x: float(x), features))
            features = np.array(features)

            assert len(features) == 4096
            npy_path = os.path.join(video_path, f'frame{frame_idx:04d}.npy')

            with open(npy_path, 'wb') as npy_f:
                np.save(npy_f, features)

            line = f.readline()

    print('\nParse complete')


if __name__ == '__main__':
    # extract_features_2d_cnn(
    #     'D:/ML Dataset/MSVD/annotations.txt',
    #     'D:/ML Dataset/MSVD/YouTubeClips',
    #     'D:/ML Dataset/MSVD/new_extracted',
    #     batch_size=8,
    # )
    extract_features_3d_cnn(
        'D:/ML Dataset/MSVD/annotations.txt',
        'D:/ML Dataset/MSVD/YouTubeClips',
        'D:/ML Dataset/MSVD/new_extracted',
        model_name='shufflenet',
    )
