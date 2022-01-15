import torch

EOS_TAG = '<EOS>'
BOS_TAG = '<BOS>'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VGG_MEAN = [103.939, 116.779, 123.68]
VGG_STD = [1, 1, 1]