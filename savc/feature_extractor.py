import os
import torch
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, Normalize, Resize
from savc.dataset import CNNExtractedMSVD, RawMSVDDataset, SequenceImageMSVDDataset
from savc.models.shufflenet import get_model as ShuffleNet
from savc.models.shufflenetv2 import get_model as ShuffleNetV2
from savc.models.resnext import get_model as ResNext101
from savc.models.sdn import SDN
from constant import *
from utils import build_video_dict


def extract_features_2d_cnn(annotations_file: str, root_path: str, output_dir: str, batch_size: int = 32) -> None:
    os.makedirs(output_dir, exist_ok=True)
    regnet_y_32gf = models.regnet_y_32gf(pretrained=True).to(DEVICE)
    # remove last layer
    regnet_y_32gf.fc = torch.nn.Identity()
    regnet_y_32gf.eval()

    all_videos = os.listdir(root_path)
    video_dict = build_video_dict(annotations_file)
    preprocess_funcs = [Normalize(IMAGE_MEAN, IMAGE_STD)]

    for idx, video in enumerate(all_videos):
        print(f'Extracting video {idx+1}/{len(all_videos)}', end='\r')
        video_index = video_dict[video]

        raw_dataset = RawMSVDDataset(os.path.join(root_path, video), preprocess_funcs)
        data_loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False)

        res = None
        total_frames = 0
        for _, (X, _) in enumerate(data_loader):
            X = X.to(DEVICE)
            out = regnet_y_32gf(X)
            total_frames += out.shape[0]

            out = torch.sum(out, dim=0)
            out_numpy = out.cpu().detach().numpy()

            if res is None:
                res = out_numpy
            else:
                res += out_numpy

        res /= total_frames
        npy_path = os.path.join(output_dir, f'video{video_index:04d}_cnn_2d_regnety32.npy')
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
            './checkpoints/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth',
            width_mult=2.,
        )
    elif model_name == 'shufflenet':
        model = ShuffleNet(
            './checkpoints/shufflenet/kinetics_shufflenet_2.0x_G3_RGB_16_best.pth',
            width_mult=2.,
        )
    elif model_name == 'resnext101':
        model = ResNext101('./checkpoints/resnext101/kinetics_resnext_101_RGB_16_best.pth')
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


def extract_semantics(
    model_path: str,
    annotation_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 20,
    start_idx: int = 0,
):
    checkpoint = torch.load(model_path)
    cnn_2d_model = checkpoint['cnn_2d_model']
    cnn_3d_model = checkpoint['cnn_3d_model']

    dataset = CNNExtractedMSVD(annotation_path, data_path, 300, cnn_2d_model, cnn_3d_model)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    dataloader_len = len(dataloader)

    model = SDN(
        cnn_features_size=CNN_3D_FEATURES_SIZE[cnn_3d_model] + CNN_2D_FEATURES_SIZE[cnn_2d_model],
        num_tags=len(dataset.tag_dict),
        dropout_rate=0.6,
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(output_path, exist_ok=True)

    for batch_idx, (X, _, _) in enumerate(dataloader):
        print(f'Extracting semantics {batch_idx+1}/{dataloader_len}', end='\r')

        X = X.to(DEVICE)
        out = model(X)
        out = (out >= 0.5).type(torch.FloatTensor)

        out_numpy = out.cpu().detach().numpy()

        for i in range(out_numpy.shape[0]):
            npy_path = os.path.join(output_path, f'video{(batch_idx*batch_size)+i+start_idx:04d}_semantic_features.npy')
            with open(npy_path, 'wb') as f:
                np.save(f, out_numpy[i])

    print('\nExtract semantic features complete')


if __name__ == '__main__':
    # extract_features_2d_cnn(
    #     'D:/ML Dataset/MSVD/annotations.txt',
    #     'D:/ML Dataset/MSVD/YouTubeClips',
    #     'D:/ML Dataset/MSVD/new_extracted/regnety',
    #     batch_size=8,
    # )
    # extract_features_3d_cnn(
    #     'D:/ML Dataset/MSVD/annotations.txt',
    #     'D:/ML Dataset/MSVD/YouTubeClips',
    #     'D:/ML Dataset/MSVD/new_extracted/resnext101_with_std',
    #     model_name='resnext101',
    # )

    cnn2d_model = ['regnety32', 'regnetx32', 'vgg']
    cnn3d_model = ['resnext101', 'shufflenet', 'shufflenetv2']
    data_type = ['train', 'test', 'validation', 'train_val']

    for model_2d in cnn2d_model:
        for model_3d in cnn3d_model:
            for type in data_type:
                if type == 'train':
                    start_idx = 0
                elif type == 'validation':
                    start_idx = 1200
                else:
                    start_idx = 1300
                # combine_cnn_features(
                #     f'D:/ML Dataset/MSVD/new_extracted/{type}',
                #     f'D:/ML Dataset/MSVD/features/{model_2d}_{model_3d}/cnn/{type}',
                #     model_2d,
                #     model_3d,
                # )
                extract_semantics(
                    f'./checkpoints/sdn/{model_2d}_{model_3d}_best.pth',
                    'D:/ML Dataset/MSVD/annotations.txt',
                    f'D:/ML Dataset/MSVD/new_extracted/{type}',
                    f'D:/ML Dataset/MSVD/features/{model_2d}_{model_3d}/semantics',
                    start_idx=start_idx,
                )
