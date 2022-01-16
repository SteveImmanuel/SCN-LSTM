import torch
import argparse
from torchvision.transforms import RandomCrop, Normalize
from torch.utils.data import DataLoader
from s2vt.dataset import MSVDDataset
from s2vt.model import S2VT
from s2vt.constant import *
from helper import *

parser = argparse.ArgumentParser(description='S2VT Implementation using PyTorch')
parser.add_argument('--annotation-path', help='File path to annotation', required=True)
parser.add_argument('--train-data-dir', help='Directory path to training annotation', required=True)
parser.add_argument('--val-data-dir', help='Directory path to training images', required=True)
parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
parser.add_argument('--learning-rate', help='Learning rate for training', default=1e-4, type=float)
parser.add_argument('--model-path', help='Load pretrained model')
parser.add_argument('--ckpt-path', help='Checkpoint path, will save for each epoch')
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
print('\n######### TRAINING CONFIGURATION #########')
print('Annotation file:', annotation_path)
print('Training directory:', train_data_dir)
print('Validation directory:', val_data_dir)
print('Batch size:', batch_size)
print('Epoch:', epoch)
print('Learning rate:', learning_rate)
print('Model path:', model_path)
print('Test overfit:', test_overfit)

# prepare train and validation dataset
train_dataset = MSVDDataset(
    train_data_dir,
    annotation_path,
    transform=[Normalize(VGG_MEAN, VGG_STD), RandomCrop(227)],
    timestep=timestep,
)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

val_dataset = MSVDDataset(
    val_data_dir,
    annotation_path,
    transform=[Normalize(VGG_MEAN, VGG_STD), RandomCrop(227)],
    timestep=timestep,
)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

# create and prepare model
model = S2VT(
    word_to_idx=train_dataset.word_to_idx,
    vocab_size=train_dataset.vocab_size,
    timestep=timestep,
).to(DEVICE)

if model_path and not test_overfit:
    model.load_state_dict(torch.load('./checkpoints/test.pt'))
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')

if test_overfit:
    print('\nTest Overfit with Small Dataset')
    print('To pass this test, you should see a very small loss on the last epoch\n')
    (X, seq_len), (y, y_mask) = next(iter(train_dataloader))
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    y_mask = y_mask.to(DEVICE)

    for epoch_idx in range(epoch):
        out = model((X, seq_len), y)

        batch_loss = torch.zeros(len(out)).to(DEVICE)
        for i in range(len(out)):
            loss = loss_func(out[i], y[i])
            loss *= y_mask[i]
            loss = torch.sum(loss).to(DEVICE)
            batch_loss[i] = loss
        batch_loss = torch.mean(batch_loss).to(DEVICE)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        print_test_overfit(batch_loss.item(), epoch_idx + 1, epoch)

else:
    train_dataloader_len = len(train_dataloader)
    val_dataloader_len = len(val_dataloader)

    for epoch_idx in range(epoch):
        print(f'\n######### Epoch-{epoch_idx+1} #########')
        train_batch_losses = torch.zeros(train_dataloader_len).to(DEVICE)
        val_batch_losses = torch.zeros(val_dataloader_len).to(DEVICE)

        print('Training Phase')
        for batch_idx, ((X, seq_len), (y, y_mask)) in enumerate(train_dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y_mask = y_mask.to(DEVICE)
            out = model((X, seq_len), y)

            batch_loss = torch.zeros(len(out)).to(DEVICE)
            for i in range(len(out)):
                loss = loss_func(out[i], y[i])
                loss *= y_mask[i]
                loss = torch.sum(loss).to(DEVICE)
                batch_loss[i] = loss
            batch_loss = torch.mean(batch_loss).to(DEVICE)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_batch_losses[batch_idx] = batch_loss.item()
            print_batch_loss(batch_loss.item(), batch_idx + 1, train_dataloader_len)

        print('\nValidation Phase')
        for batch_idx, ((X, seq_len), (y, y_mask)) in enumerate(val_dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y_mask = y_mask.to(DEVICE)
            out = model((X, seq_len), y)

            batch_loss = torch.zeros(len(out)).to(DEVICE)
            for i in range(len(out)):
                loss = loss_func(out[i], y[i])
                loss *= y_mask[i]
                loss = torch.sum(loss).to(DEVICE)
                batch_loss[i] = loss
            batch_loss = torch.mean(batch_loss).to(DEVICE)

            val_batch_losses[batch_idx] = batch_loss.item()
            print_batch_loss(batch_loss.item(), batch_idx + 1, val_dataloader_len)

        avg_train_loss = torch.mean(train_batch_losses).item()
        avg_val_loss = torch.mean(val_batch_losses).item()
        print(f'Train Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')

# python train.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --train-data-dir "D:/ML Dataset/MSVD/YouTubeClips/train " --val-data-dir "D:/ML Dataset/MSVD/YouTubeClips/validation" --batch-size 8 --epoch 20 --learning-rate 1e-4