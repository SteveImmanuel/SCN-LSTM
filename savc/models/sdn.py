import math
import torch
import time
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from savc.dataset import CNNExtractedMSVD
from constant import *
from helper import *


class SDN(torch.nn.Module):
    """
    Semantic Detection Network, to learn tags from cnn features
    """
    def __init__(self, cnn_features_size: int, num_tags: int = 750, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(cnn_features_size, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, num_tags),
            torch.nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propagate

        Args:
            X (torch.Tensor): (BATCH_SIZE, cnn_features)
        Returns:
            multi label class score (BATCH_SIZE, num_tags)
        """
        return self.classifier(X)


def sdn_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Calculate loss using cross entropy of multi class loss

    Args:
        pred (torch.Tensor): [description] (BATCH_SIZE, num_tags)
        truth (torch.Tensor): [description] (BATCH_SIZE, num_tags)

    Returns:
        torch.Tensor: [description] (1, )
    """
    epsilon = 1e-20
    loss = (truth * torch.log(pred + epsilon) + (1 - truth) * torch.log(1 - pred + epsilon)) * -1
    loss = torch.sum(loss, dim=1)
    loss = torch.mean(loss, dim=0)
    return loss


def get_map(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Calculate mean average precision

    Args:
        pred (torch.Tensor): [description] (BATCH_SIZE, num_tags)
        truth (torch.Tensor): [description] (BATCH_SIZE, num_tags)

    Returns:
        torch.Tensor: [description]
    """
    truth = truth.type(torch.LongTensor)
    pred = (pred >= 0.5).type(torch.LongTensor)
    acc = (truth == pred).type(torch.FloatTensor)
    return torch.mean(acc)


if __name__ == '__main__':
    uid = int(time.time())
    annotation_path = 'D:/ML Dataset/MSVD/annotations.txt'
    train_path = 'D:/ML Dataset/MSVD/new_extracted/train'
    val_path = 'D:/ML Dataset/MSVD/new_extracted/validation'
    cnn_2d_model = 'regnetx32'
    cnn_3d_model = 'shufflenetv2'
    batch_size = 20
    epoch = 100
    learning_rate = 5e-3
    ckpt_dir = './checkpoints/sdn'

    # prepare train and validation dataset
    train_dataset = CNNExtractedMSVD(annotation_path, train_path, 300, cnn_2d_model, cnn_3d_model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train_dataloader_len = len(train_dataloader)

    val_dataset = CNNExtractedMSVD(annotation_path, val_path, 300, cnn_2d_model, cnn_3d_model)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader_len = len(val_dataloader)

    # create and prepare model
    model = SDN(
        cnn_features_size=4568,  # 4568 for shufflenetv2, 4440 for shufflenet
        num_tags=len(train_dataset.tag_dict),
        dropout_rate=0.6,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7, verbose=True)
    loss_func = sdn_loss

    best_val_loss = math.inf

    for epoch_idx in range(epoch):
        print(f'\n######### Epoch-{epoch_idx+1} #########')
        train_batch_losses = torch.zeros(train_dataloader_len).to(DEVICE)
        val_batch_losses = torch.zeros(val_dataloader_len).to(DEVICE)
        train_batch_map = torch.zeros(train_dataloader_len).to(DEVICE)
        val_batch_map = torch.zeros(val_dataloader_len).to(DEVICE)

        print('###Training Phase###')
        model.train()
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            out = model(X)

            batch_loss = loss_func(out, y)
            map = get_map(out, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_batch_losses[batch_idx] = batch_loss.item()
            train_batch_map[batch_idx] = map.item()
            print_batch_loss(batch_loss.item(), batch_idx + 1, train_dataloader_len)

        print('\n###Validation Phase###')
        model.eval()
        for batch_idx, (X, y) in enumerate(val_dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            out = model(X)

            batch_loss = loss_func(out, y)
            map = get_map(out, y)

            val_batch_losses[batch_idx] = batch_loss.item()
            val_batch_map[batch_idx] = map.item()
            print_batch_loss(batch_loss.item(), batch_idx + 1, val_dataloader_len)

        avg_train_loss = torch.mean(train_batch_losses).item()
        avg_val_loss = torch.mean(val_batch_losses).item()
        avg_train_map = torch.mean(train_batch_map).item()
        avg_val_map = torch.mean(val_batch_map).item()

        print(f'\nTrain Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')
        print(f'Train mAP: {avg_train_map:.5f}, Validation mAP: {avg_val_map:.5f}')
        lr_scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            filename = f'{uid}_{cnn_2d_model}_{cnn_3d_model}_best.pth'
            filepath = os.path.join(ckpt_dir, filename)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))
            print(f'Model saved to {filepath}')