import torch
import argparse
from torchvision.transforms import RandomCrop, Normalize
from torch.utils.data import DataLoader
from s2vt.utils import *
from s2vt.dataset import MSVDDataset
from s2vt.model import S2VT
from s2vt.constant import *

parser = argparse.ArgumentParser(description='S2VT Implementation using PyTorch')
parser.add_argument('--annotation-path', help='File path to annotation', required=True)
parser.add_argument('--train-data-dir', help='Directory path to training annotation', required=True)
parser.add_argument('--val-data-dir', help='Directory path to training images', required=True)
parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
parser.add_argument('--learning-rate', help='Learning rate for training', default=1e-5, type=float)
parser.add_argument('--model-path', help='Load pretrained model')
parser.add_argument(
    '--test-overfit',
    help='Sanity check to test overfit model with very small dataset',
    action='store_true',
)

args = parser.parse_args()
annotation_path = args.annotation_path
train_data_dir = args.train_data_dir
val_data_dir = args.val_data_dir
timestep = args.timestep
batch_size = args.batch_size
epoch = args.epoch
learning_rate = args.learning_rate
model_path = args.model_path
test_overfit = args.test_overfit

# show training config
print('TRAINING CONFIGURATION')
print('Annotation file:', annotation_path)
print('Training directory:', train_data_dir)
print('Validation directory:', val_data_dir)
print('Batch size:', batch_size)
print('Epoch:', epoch)
print('Learning rate:', learning_rate)
print('Model path:', model_path)
print('Test overfit:', test_overfit)

dataset = MSVDDataset(
    'D:/ML Dataset/MSVD/YouTubeClips',
    'D:/ML Dataset/MSVD/annotations.txt',
    transform=[Normalize(VGG_MEAN, VGG_STD), RandomCrop(227)],
    timestep=timestep,
)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
model = S2VT(vocab_size=dataset.get_vocab_size(), timestep=timestep).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')

(X, seq_len), (y, y_mask) = next(iter(dataloader))
for i in range(epoch):
    # for (X, seq_len), (y, y_mask) in dataloader:
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    y_mask = y_mask.to(DEVICE)
    model.train()
    # model.eval()
    out = model((X, seq_len), y)

    batch_loss = torch.zeros(len(out))

    for i in range(len(out)):
        loss = loss_func(out[i], y[i])
        loss *= y_mask[i]
        loss = torch.sum(loss)
        batch_loss[i] = loss
    batch_loss = torch.mean(batch_loss)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    print(batch_loss)
torch.save(model.state_dict(), './checkpoints/test.pt')