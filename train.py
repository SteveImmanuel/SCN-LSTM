import os
import argparse
import torch
import time
import traceback
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from s2vt.utils import idx_to_annotation
from s2vt.dataset import PreprocessedMSVDDataset
from s2vt.model import S2VT
from s2vt.constant import *
from helper import *


def run():
    parser = argparse.ArgumentParser(description='Train S2VT Model')
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
    print('Timestep:', timestep)
    print('Epoch:', epoch)
    print('Learning rate:', learning_rate)
    print('Test overfit:', test_overfit)

    # prepare train and validation dataset
    train_dataset = PreprocessedMSVDDataset(
        train_data_dir,
        annotation_path,
        timestep=timestep,
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    val_dataset = PreprocessedMSVDDataset(
        val_data_dir,
        annotation_path,
        timestep=timestep,
    )
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    # create and prepare model
    model = S2VT(
        word_to_idx=train_dataset.word_to_idx,
        vocab_size=train_dataset.vocab_size,
        timestep=timestep,
        lstm_hidden_size=500,
        drop_out_rate=0.3,
    ).to(DEVICE)

    if model_path and not test_overfit:
        print(f'\nLoading pretrained model in {model_path}\n')
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7, verbose=True)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    if test_overfit:
        print('\nTest Overfit with Small Dataset')
        print('To pass this test, you should see a very small loss on the last epoch\n')
        X, (y, y_mask) = next(iter(train_dataloader))
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y_mask = y_mask.to(DEVICE)

        model.train()
        for epoch_idx in range(epoch):
            out = model(X, y)

            out_flat = out.view(-1, train_dataset.vocab_size)
            y_flat = y[:, 1:].contiguous().view(-1)
            y_mask_flat = y_mask[:, 1:].contiguous().view(-1)

            batch_loss = loss_func(out_flat, y_flat)
            batch_loss = batch_loss * y_mask_flat
            batch_loss = torch.sum(batch_loss) / torch.sum(y_mask_flat)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print_test_overfit(batch_loss.item(), epoch_idx + 1, epoch)

        model.eval()
        out = model(X)
        out = torch.argmax(out, dim=2).to(DEVICE).long()
        res = idx_to_annotation(out[0].tolist(), train_dataset.idx_to_word)
        grount_truth = idx_to_annotation(y[0][1:].tolist(), train_dataset.idx_to_word)
        print('\nPrediction:', format_result(res))
        print('Ground Truth:', format_result(grount_truth))

    else:
        uid = int(time.time())
        try:
            batch_loss_log_path = os.path.join(log_dir, f'{uid}_batch_loss.csv')
            epoch_loss_log_path = os.path.join(log_dir, f'{uid}_epoch_loss.csv')
            batch_loss_log = create_batch_log_file(batch_loss_log_path)
            epoch_loss_log = create_epoch_log_file(epoch_loss_log_path)
            best_val_loss = math.inf

            train_dataloader_len = len(train_dataloader)
            val_dataloader_len = len(val_dataloader)

            for epoch_idx in range(epoch):
                print(f'\n######### Epoch-{epoch_idx+1} #########')
                train_batch_losses = torch.zeros(train_dataloader_len).to(DEVICE)
                val_batch_losses = torch.zeros(val_dataloader_len).to(DEVICE)

                print('###Training Phase###')
                model.train()
                for batch_idx, (X, (y, y_mask)) in enumerate(train_dataloader):
                    X = X.to(DEVICE)
                    y = y.to(DEVICE)
                    y_mask = y_mask.to(DEVICE)
                    out = model(X, y)

                    out = out.view(-1, train_dataset.vocab_size)
                    y = y[:, 1:].contiguous().view(-1)
                    y_mask = y_mask[:, 1:].contiguous().view(-1)

                    batch_loss = loss_func(out, y)
                    batch_loss = batch_loss * y_mask
                    batch_loss = torch.sum(batch_loss) / torch.sum(y_mask)

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    train_batch_losses[batch_idx] = batch_loss.item()
                    print_batch_loss(batch_loss.item(), batch_idx + 1, train_dataloader_len)
                    batch_loss_log.write(f'{epoch_idx*train_dataloader_len+batch_idx},{batch_loss.item()}\n')
                    batch_loss_log.flush()

                print('\n###Validation Phase###')
                for batch_idx, (X, (y, y_mask)) in enumerate(val_dataloader):
                    X = X.to(DEVICE)
                    y = y.to(DEVICE)
                    y_mask = y_mask.to(DEVICE)
                    out = model(X, y)

                    temp_X = X[0:1]
                    temp_y = y[0]

                    out = out.view(-1, train_dataset.vocab_size)
                    y = y[:, 1:].contiguous().view(-1)
                    y_mask = y_mask[:, 1:].contiguous().view(-1)

                    batch_loss = loss_func(out, y)
                    batch_loss = batch_loss * y_mask
                    batch_loss = torch.sum(batch_loss) / torch.sum(y_mask)

                    val_batch_losses[batch_idx] = batch_loss.item()
                    print_batch_loss(batch_loss.item(), batch_idx + 1, val_dataloader_len)

                avg_train_loss = torch.mean(train_batch_losses).item()
                avg_val_loss = torch.mean(val_batch_losses).item()
                print(f'\nTrain Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')
                epoch_loss_log.write(f'{epoch_idx},{avg_train_loss},{avg_val_loss}\n')
                epoch_loss_log.flush()

                model.eval()
                temp_out = model(temp_X)[0]
                temp_out = torch.argmax(temp_out, dim=1).to(DEVICE).long()
                temp_out = idx_to_annotation(temp_out.tolist(), val_dataset.idx_to_word)
                temp_y = idx_to_annotation(temp_y.tolist(), val_dataset.idx_to_word)
                print('###Current Result###')
                print('Prediction:', format_result(temp_out))
                print('Ground Truth:', format_result(temp_y[1:]))

                # save model checkpoint
                if ckpt_dir:
                    if (epoch_idx % ckpt_interval == 0 or epoch_idx == epoch - 1):
                        filename = f'{uid}_epoch{epoch_idx:03}_{avg_train_loss:.3f}_{avg_val_loss:.3f}.pth'
                        filepath = os.path.join(ckpt_dir, filename)
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))
                        print(f'Model saved to {filepath}')

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        filename = f'{uid}_best_weights.pth'
                        filepath = os.path.join(ckpt_dir, filename)
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))
                        print(f'Model saved to {filepath}')

                lr_scheduler.step(avg_val_loss)

        except Exception:
            traceback.print_exc()
            if ckpt_dir:
                print('Error occured, saving current progress')
                filename = f'{uid}_backup_error.pth'
                filepath = os.path.join(ckpt_dir, filename)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))
                print(f'Model saved to {filepath}')

        finally:
            batch_loss_log.close()
            epoch_loss_log.close()


if __name__ == '__main__':
    run()

# python train.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --train-data-dir "C:/MSVD_extracted/out/train" --val-data-dir "C:/MSVD_extracted/out/validation" --batch-size 10 --epoch 100 --learning-rate 1e-4 --ckpt-interval 10