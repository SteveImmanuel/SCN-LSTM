import os
import argparse
import torch
import time
from datetime import datetime
from torchvision.transforms import RandomCrop, Normalize
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from s2vt.dataset import MSVDDataset
from s2vt.model import S2VT
from s2vt.constant import *
from helper import *

parser = argparse.ArgumentParser(description='S2VT Implementation using PyTorch')
parser.add_argument('--annotation-path', help='File path to annotation', required=True)
parser.add_argument('--train-data-dir', help='Directory path to training annotation', required=True)
parser.add_argument('--val-data-dir', help='Directory path to training images', required=True)
parser.add_argument('--ckpt-dir', help='Checkpoint directory, will save for each epoch', default='./checkpoints')
parser.add_argument('--ckpt-interval', help='How many epoch between checkpoints', default=1, type=int)
parser.add_argument('--log-dir', help='Log directory', default='./logs')
parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
parser.add_argument('--learning-rate', help='Learning rate for training', default=1e-4, type=float)
parser.add_argument('--momentum', help='Momentum for updating gradient', default=9e-1, type=float)
parser.add_argument('--gamma', help='Gamma for learning rate scheduler', default=5e-1, type=float)
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
momentum = args.momentum
gamma = args.gamma
model_path = args.model_path
test_overfit = args.test_overfit
ckpt_dir = args.ckpt_dir
ckpt_interval = args.ckpt_interval
log_dir = args.log_dir

# show training config
print('\n######### TRAINING CONFIGURATION #########')
print('Annotation file:', annotation_path)
print('Training directory:', train_data_dir)
print('Validation directory:', val_data_dir)
print('Checkpoint directory:', ckpt_dir)
print('Checkpoint interval:', ckpt_interval)
print('Log directory:', log_dir)
print('Pretrained model path:', model_path)
print('Batch size:', batch_size)
print('Epoch:', epoch)
print('Learning rate:', learning_rate)
print('Momentum:', momentum)
print('Gamma:', gamma)
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
    drop_out_rate=0.1,
).to(DEVICE)

if model_path and not test_overfit:
    print(f'\nLoading pretrained model in {model_path}\n')
    model.load_state_dict(torch.load(model_path))
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
lr_scheduler = StepLR(optimizer, step_size=20, gamma=gamma)
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
    try:
        uid = int(time.time())
        batch_loss_log_path = os.path.join(log_dir, f'{uid}_batch_loss.csv')
        epoch_loss_log_path = os.path.join(log_dir, f'{uid}_epoch_loss.csv')
        batch_loss_log = create_batch_log_file(batch_loss_log_path)
        epoch_loss_log = create_epoch_log_file(epoch_loss_log_path)

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
                batch_loss_log.write(f'{epoch_idx*train_dataloader_len+batch_idx},{batch_loss.item()}\n')
                batch_loss_log.flush()

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
            print(f'\nTrain Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')
            epoch_loss_log.write(f'{epoch_idx},{avg_train_loss},{avg_val_loss}\n')
            epoch_loss_log.flush()

            # save model checkpoint
            if ckpt_dir and (epoch_idx % ckpt_interval == 0 or epoch_idx == epoch - 1):
                timestamp = datetime.strftime(datetime.now(), '%d-%H-%M')
                filename = f'{timestamp}_{avg_train_loss:.3f}_{avg_val_loss:.3f}.pth'
                filepath = os.path.join(ckpt_dir, filename)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))
                print(f'Model saved to {filepath}')

            lr_scheduler.step()

    except Exception as e:
        print(e)
        if ckpt_dir:
            print('Error occured, saving current progress')
            timestamp = datetime.strftime(datetime.now(), '%d-%H-%M')
            filename = f'{timestamp}_backup_error.pth'
            filepath = os.path.join(ckpt_dir, filename)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))
            print(f'Model saved to {filepath}')

    finally:
        batch_loss_log.close()
        epoch_loss_log.close()

# python train.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --train-data-dir "D:/ML Dataset/MSVD/YouTubeClips/train" --val-data-dir "D:/ML Dataset/MSVD/YouTubeClips/validation" --batch-size 8 --epoch 5 --learning-rate 1e-3

# python train.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --train-data-dir "D:/ML Dataset/MSVD/YouTubeClips/train" --val-data-dir "D:/ML Dataset/MSVD/YouTubeClips/validation" --batch-size 8 --epoch 10 --learning-rate 1e-3 --momentum 0.9 --gamma 0.5