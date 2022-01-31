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


def combine_cnn_features(input_dir: str, output_dir: str, cnn_2d_model: str, cnn_3d_model: str):
    os.makedirs(output_dir, exist_ok=True)
    videos = list(set(map(lambda x: x[5:9], os.listdir(input_dir))))

    for idx, video_idx in enumerate(videos):
        print(f'Combining {idx+1}/{len(videos)}', end='\r')

        cnn2d_npy_path = os.path.join(input_dir, f'video{video_idx}_cnn_2d_{cnn_2d_model}.npy')
        cnn3d_npy_path = os.path.join(input_dir, f'video{video_idx}_cnn_3d_{cnn_3d_model}.npy')
        cnn2d_features = np.load(cnn2d_npy_path)
        cnn3d_features = np.load(cnn3d_npy_path)
        features = np.concatenate((cnn3d_features, cnn2d_features), axis=0)

        out_path = os.path.join(output_dir, f'video{video_idx}_cnn_features.npy')
        with open(out_path, 'wb') as f:
            np.save(f, features)

    print('\nCombine features complete')


if __name__ == '__main__':
    # extract_features_2d_cnn(
    #     'D:/ML Dataset/MSVD/annotations.txt',
    #     'D:/ML Dataset/MSVD/YouTubeClips',
    #     'D:/ML Dataset/MSVD/new_extracted',
    #     batch_size=8,
    # )
    # extract_features_3d_cnn(
    #     'D:/ML Dataset/MSVD/annotations.txt',
    #     'D:/ML Dataset/MSVD/YouTubeClips',
    #     'D:/ML Dataset/MSVD/new_extracted',
    #     model_name='shufflenet',
    # )
    cnn2d_model = 'regnetx32'
    cnn3d_model = 'shufflenetv2'

    combine_cnn_features(
        'D:/ML Dataset/MSVD/new_extracted/train',
        f'D:/ML Dataset/MSVD/combined_feature/{cnn2d_model}_{cnn3d_model}',
        cnn2d_model,
        cnn3d_model,
    )
