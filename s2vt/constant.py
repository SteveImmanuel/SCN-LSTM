import torch

UNKNOWN_TAG = '<UNK>'
PAD_TAG = '<PAD>'
EOS_TAG = '<EOS>'
BOS_TAG = '<BOS>'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://pytorch.org/hub/pytorch_vision_vgg/
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]