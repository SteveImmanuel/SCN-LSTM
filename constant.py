import torch

UNKNOWN_TAG = '<UNK>'
PAD_TAG = '<PAD>'
EOS_TAG = '<EOS>'
BOS_TAG = '<BOS>'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://pytorch.org/hub/pytorch_vision_vgg/
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

KINETICS_MEAN = [110.63666788, 103.16065604, 96.29023126]
KINETICS_STD = [1, 1, 1]

CNN_3D_FEATURES_SIZE = {
    'shufflenet': 1920,
    'shufflenetv2': 2048,
}

CNN_2D_FEATURES_SIZE = {
    'vgg': 4096,
    'regnetx32': 2520,
}

SEMANTIC_SIZE = 300
