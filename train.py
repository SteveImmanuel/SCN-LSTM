import torch
from torchvision.transforms import RandomCrop
from torch.utils.data import DataLoader
from s2vt.utils import *
from s2vt.dataset import MSVDDataset
from s2vt.model import S2VT

dataset = MSVDDataset(
    'D:/ML Dataset/MSVD/YouTubeClips',
    'D:/ML Dataset/MSVD/annotations.txt',
    transform=[RandomCrop(227)],
)
dataloader = DataLoader(dataset)
model = S2VT(vocab_size=dataset.get_vocab_size()).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')

for i in range(10):
    for X, (y, y_mask) in dataloader:
        X = X.cuda()
        y = y.cuda()
        y_mask = y_mask.cuda()

        out = model(X, y)

        batch_loss = torch.zeros(dataloader.batch_size)
        for i in range(dataloader.batch_size):
            loss = loss_func(out, y)
            loss *= y_mask
            loss = torch.sum(loss)
            batch_loss[i] = loss
        batch_loss = torch.mean(batch_loss)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        print(batch_loss)