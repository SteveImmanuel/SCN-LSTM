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
KINETICS_STD = [38.7568578, 37.88248729, 40.02898126]

CNN_3D_FEATURES_SIZE = {
    'shufflenet': 1920,
    'shufflenetv2': 2048,
    'resnext101': 2048,
    'eco': 1536,
}

CNN_2D_FEATURES_SIZE = {
    'vgg': 4096,
    'regnetx32': 2520,
    'regnety32': 3712,
    'resnext': 2048,
}

SEMANTIC_SIZE = 300

BLACKLIST_WORDS = [
    'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around', 'as', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning',
    'considering', 'despite', 'down', 'up', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from',
    'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past',
    'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards', 'under',
    'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without', 'a', 'an', 'is', 'the',
    'are', 'were', 'was', 'his', 'her', 's', 'out', 'it'
]
