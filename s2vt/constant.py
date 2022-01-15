import torch

EOS_TAG = '<EOS>'
BOS_TAG = '<BOS>'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')