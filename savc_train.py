import os
import argparse
import torch
import time
import traceback
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils import generate_epsilon, idx_to_annotation
from savc.dataset import CompiledMSVD, ExtractedMSVD
from savc.models.scn import SemanticLSTM
from constant import *
from helper import *


def run():
    parser = argparse.ArgumentParser(description='Train S2VT Model')
    parser.add_argument('--annotation-path', help='File path to annotation', required=True)
    parser.add_argument('--dataset-dir', help='Directory path to dataset for train and validation', required=True)
    parser.add_argument('--ckpt-dir', help='Checkpoint directory, will save for each epoch', default='./checkpoints')
    parser.add_argument('--ckpt-interval', help='How many epoch between checkpoints', default=1, type=int)
    parser.add_argument('--log-dir', help='Log directory', default='./logs')
    parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
    parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    parser.add_argument('--learning-rate', help='Learning rate for training', default=1e-4, type=float)
    parser.add_argument('--model-path', help='Load pretrained model')
    parser.add_argument(
        '--model-cnn-2d',
        help='2D CNN model architecture',
        choices=['vgg', 'regnetx32', 'regnety32', 'resnext'],
        required=True,
    )
    parser.add_argument(
        '--model-cnn-3d',
        help='3D CNN model architecture',
        choices=['shufflenet', 'shufflenetv2', 'resnext101', 'eco'],
        required=True,
    )
    parser.add_argument(
        '--test-overfit',
        help='Sanity check to test overfit model with very small dataset',
        action='store_true',
    )
    parser.add_argument(
        '--mode',
        help='Use sample distribution or argmax',
        choices=['sample', 'argmax'],
        required=True,
    )

    args = parser.parse_args()
    annotation_path = args.annotation_path
    model_cnn_2d = args.model_cnn_2d
    model_cnn_3d = args.model_cnn_3d
    dataset_dir = args.dataset_dir
    timestep = args.timestep
    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.learning_rate
    model_path = args.model_path
    test_overfit = args.test_overfit
    ckpt_dir = args.ckpt_dir
    ckpt_interval = args.ckpt_interval
    log_dir = args.log_dir
    mode = args.mode

    # show training config
    print('\n######### TRAINING CONFIGURATION #########')
    print('Annotation file:', annotation_path)
    print('2D CNN model:', model_cnn_2d)
    print('3D CNN model:', model_cnn_3d)
    print('Generate mode:', mode)
    print('Dataset directory:', dataset_dir)
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
    # train_dataset = CompiledMSVD(
    #     annotation_path,
    #     os.path.join(dataset_dir, f'{model_cnn_2d}_{model_cnn_3d}', 'cnn', 'train_val'),
    #     os.path.join(dataset_dir, f'{model_cnn_2d}_{model_cnn_3d}', 'semantics', 'train_val'),
    #     timestep=timestep,
    #     beta=0.5,
    # )
    train_dataset = ExtractedMSVD(
        annotation_path,
        os.path.join(dataset_dir, 'msvd_resnext_eco.npy'),
        os.path.join(dataset_dir, 'msvd_tag_gt_4_msvd.npy'),
        timestep=timestep,
        max_sentence_len=25,
        beta=0.7,
        type='train',
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # val_dataset = CompiledMSVD(
    #     annotation_path,
    #     os.path.join(dataset_dir, f'{model_cnn_2d}_{model_cnn_3d}', 'cnn', 'testing'),
    #     os.path.join(dataset_dir, f'{model_cnn_2d}_{model_cnn_3d}', 'semantics', 'testing'),
    #     timestep=timestep,
    #     beta=7,
    #     max_len=100,
    # )
    val_dataset = ExtractedMSVD(
        annotation_path,
        os.path.join(dataset_dir, 'msvd_resnext_eco.npy'),
        os.path.join(dataset_dir, 'msvd_tag_gt_4_msvd.npy'),
        timestep=timestep,
        max_sentence_len=25,
        beta=0.7,
        type='test',
        max_len=100,
    )
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    # create and prepare model
    model = SemanticLSTM(
        cnn_feature_size=CNN_3D_FEATURES_SIZE[model_cnn_3d] + CNN_2D_FEATURES_SIZE[model_cnn_2d],
        vocab_size=train_dataset.vocab_size,
        semantic_size=SEMANTIC_SIZE,
        hidden_size=512,
        input_size=512,
        embed_size=300,
        timestep=timestep,
        drop_out_rate=0.5,
    ).to(DEVICE)

    if model_path and not test_overfit:
        checkpoint = torch.load(model_path)
        print(f'\nLoading pretrained model in {model_path}\n')
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7, verbose=True)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    if test_overfit:
        print('\nTest Overfit with Small Dataset')
        print('To pass this test, you should see a very small loss on the last epoch\n')
        cnn_features, semantic_features, cap, cap_mask = next(iter(train_dataloader))
        cnn_features = cnn_features.to(DEVICE)
        semantic_features = semantic_features.to(DEVICE)
        cap = cap.to(DEVICE)
        cap_mask = cap_mask.to(DEVICE)

        model.train()
        for epoch_idx in range(epoch):
            out = model(cap, cnn_features, semantic_features)

            out_flat = out.view(-1, train_dataset.vocab_size)
            cap_flat = cap[:, 1:].contiguous().view(-1)
            cap_mask_flat = cap_mask[:, 1:].contiguous().view(-1)

            batch_loss = loss_func(out_flat, cap_flat)
            batch_loss = batch_loss * cap_mask_flat
            batch_loss = torch.sum(batch_loss) / torch.sum(cap_mask_flat)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print_test_overfit(batch_loss.item(), epoch_idx + 1, epoch)

        model.eval()
        out = model(cap, cnn_features, semantic_features)
        out = torch.argmax(out, dim=2).to(DEVICE).long()
        res = idx_to_annotation(out[0].tolist(), train_dataset.idx_to_word)
        grount_truth = idx_to_annotation(cap[0][1:].tolist(), train_dataset.idx_to_word)
        print('\nPrediction:', format_result(res))
        print('Ground Truth:', format_result(grount_truth))

    else:
        uid = int(time.time())

        try:
            save_checkpoint = {
                'model_state_dict': None,
                'model_cnn_2d': model_cnn_2d,
                'model_cnn_3d': model_cnn_3d,
                'hidden_size': model.hidden_size,
                'input_size': model.input_size,
                'drop_out_rate': model.drop_out_rate,
                'vocab_size': model.vocab_size,
                'semantic_size': model.semantic_size,
                'cnn_feature_size': model.cnn_feature_size,
                'timestep': model.timestep,
                'embed_size': model.embed_size
            }
            epsilons = generate_epsilon(epoch)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
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
                for batch_idx, (cnn_features, semantic_features, cap, cap_mask) in enumerate(train_dataloader):
                    cnn_features = cnn_features.to(DEVICE)
                    semantic_features = semantic_features.to(DEVICE)
                    cap = cap.to(DEVICE)
                    cap_mask = cap_mask.to(DEVICE)

                    out = model(cap, cnn_features, semantic_features, epsilons[epoch_idx], mode=mode)

                    out = out.view(-1, train_dataset.vocab_size)
                    cap = cap[:, 1:].contiguous().view(-1)
                    cap_mask = cap_mask[:, 1:].contiguous().view(-1)

                    batch_loss = loss_func(out, cap)
                    batch_loss = batch_loss * cap_mask
                    batch_loss = torch.sum(batch_loss) / torch.sum(cap_mask)

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    train_batch_losses[batch_idx] = batch_loss.item()
                    print_batch_loss(batch_loss.item(), batch_idx + 1, train_dataloader_len)
                    batch_loss_log.write(f'{epoch_idx*train_dataloader_len+batch_idx},{batch_loss.item()}\n')
                    batch_loss_log.flush()

                print('\n###Validation Phase###')
                for batch_idx, (cnn_features, semantic_features, cap, cap_mask) in enumerate(val_dataloader):
                    cnn_features = cnn_features.to(DEVICE)
                    semantic_features = semantic_features.to(DEVICE)
                    cap = cap.to(DEVICE)
                    cap_mask = cap_mask.to(DEVICE)

                    out = model(cap, cnn_features, semantic_features, epsilons[epoch_idx], mode=mode)

                    temp_cnn_features = cnn_features[0:1]
                    temp_semantic_features = semantic_features[0:1]
                    temp_cap = cap[0:1]

                    out = out.view(-1, train_dataset.vocab_size)
                    cap = cap[:, 1:].contiguous().view(-1)
                    cap_mask = cap_mask[:, 1:].contiguous().view(-1)

                    batch_loss = loss_func(out, cap)
                    batch_loss = batch_loss * cap_mask
                    batch_loss = torch.sum(batch_loss) / torch.sum(cap_mask)

                    val_batch_losses[batch_idx] = batch_loss.item()
                    print_batch_loss(batch_loss.item(), batch_idx + 1, val_dataloader_len)

                avg_train_loss = torch.mean(train_batch_losses).item()
                avg_val_loss = torch.mean(val_batch_losses).item()
                print(f'\nTrain Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')
                epoch_loss_log.write(f'{epoch_idx},{avg_train_loss},{avg_val_loss}\n')
                epoch_loss_log.flush()

                model.eval()
                temp_out = model(temp_cap, temp_cnn_features, temp_semantic_features, mode=mode)[0]
                temp_out = idx_to_annotation(temp_out.tolist(), val_dataset.idx_to_word)
                temp_cap = idx_to_annotation(temp_cap[0].tolist(), val_dataset.idx_to_word)
                print('###Current Result###')
                print('Prediction:', format_result(temp_out))
                print('Ground Truth:', format_result(temp_cap[1:]))

                # save model checkpoint
                if ckpt_dir:
                    if (epoch_idx % ckpt_interval == 0 or epoch_idx == epoch - 1):
                        filename = f'{uid}_{model_cnn_2d}_{model_cnn_3d}_epoch{epoch_idx:03}_{avg_train_loss:.3f}_{avg_val_loss:.3f}.pth'
                        filepath = os.path.join(ckpt_dir, filename)
                        save_checkpoint['model_state_dict'] = model.state_dict()
                        torch.save(save_checkpoint, os.path.join(ckpt_dir, filename))
                        print(f'Model saved to {filepath}')

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        filename = f'{uid}_{model_cnn_2d}_{model_cnn_3d}_best_weights.pth'
                        filepath = os.path.join(ckpt_dir, filename)
                        save_checkpoint['model_state_dict'] = model.state_dict()
                        torch.save(save_checkpoint, os.path.join(ckpt_dir, filename))
                        print(f'Model saved to {filepath}')

                lr_scheduler.step(avg_val_loss)

        except Exception:
            traceback.print_exc()
            if ckpt_dir:
                print('Error occured, saving current progress')
                filename = f'{uid}_{model_cnn_2d}_{model_cnn_3d}_backup_error.pth'
                filepath = os.path.join(ckpt_dir, filename)
                save_checkpoint['model_state_dict'] = model.state_dict()
                torch.save(save_checkpoint, os.path.join(ckpt_dir, filename))
                print(f'Model saved to {filepath}')

        finally:
            batch_loss_log.close()
            epoch_loss_log.close()


if __name__ == '__main__':
    run()

# python savc_train.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --dataset-dir "D:/ML Dataset/MSVD/features" --batch-size 64 --epoch 150 --learning-rate 5e-4 --ckpt-interval 10 --model-cnn-2d "regnety32" --model-cnn-3d "resnext101" --ckpt-dir "./checkpoints/savc" --log-dir "./logs/savc" --mode argmax
# python savc_train.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --dataset-dir "D:/ML Dataset/MSVD/downloaded" --batch-size 64 --epoch 150 --learning-rate 5e-4 --ckpt-interval 10 --model-cnn-2d "resnext" --model-cnn-3d "eco" --ckpt-dir "./checkpoints/savc" --log-dir "./logs/savc" --mode sample --timestep 30