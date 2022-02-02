import os
import traceback
import argparse
import torch
import time
import json
from datetime import datetime
from torch.utils.data import DataLoader
from savc.dataset import CompiledMSVD
from savc.models.scn import SemanticLSTM
from utils import idx_to_annotation
from constant import *
from helper import *

parser = argparse.ArgumentParser(description='Predict using S2VT Model')
parser.add_argument('--annotation-path', help='File path to annotation', required=True)
parser.add_argument('--dataset-dir', help='Directory path to test data', required=True)
parser.add_argument('--out-path', help='Output filepath', required=True)
parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
parser.add_argument('--model-path', help='Load pretrained model', required=True)
parser.add_argument(
    '--mode',
    help='Use sample distribution or argmax',
    choices=['sample', 'argmax'],
    required=True,
)

args = parser.parse_args()
annotation_path = args.annotation_path
dataset_dir = args.dataset_dir
out_path = args.out_path
timestep = args.timestep
batch_size = args.batch_size
model_path = args.model_path
mode = args.mode

# show test config
print('\n######### TEST CONFIGURATION #########')
print('Annotation file:', annotation_path)
print('Output:', out_path)
print('Test directory:', dataset_dir)
print('Timestep:', timestep)
print('Pretrained model path:', model_path)
print('Batch size:', batch_size)

checkpoint = torch.load(model_path)
model_cnn_2d = checkpoint['model_cnn_2d']
model_cnn_3d = checkpoint['model_cnn_3d']
model_state_dict = checkpoint['model_state_dict']

# prepare train and validation dataset
test_dataset = CompiledMSVD(
    annotation_path,
    os.path.join(dataset_dir, f'{model_cnn_2d}_{model_cnn_3d}', 'cnn', 'testing'),
    os.path.join(dataset_dir, f'{model_cnn_2d}_{model_cnn_3d}', 'semantics', 'testing'),
    timestep=timestep,
)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# create and prepare model
model = SemanticLSTM(
    cnn_feature_size=CNN_3D_FEATURES_SIZE[model_cnn_3d] + CNN_2D_FEATURES_SIZE[model_cnn_2d],
    vocab_size=test_dataset.vocab_size,
    semantic_size=300,
    timestep=timestep,
    drop_out_rate=0.5,
).to(DEVICE)

print(f'\nLoading pretrained model in {model_path}\n')
model.load_state_dict(model_state_dict)
model.eval()

try:
    uid = int(time.time())

    test_dataloader_len = len(test_dataloader)
    result = {'email': 'iam.steve.immanuel@gmail.com', 'predictions': {}}
    softmax_func = torch.nn.Softmax(dim=1)

    for batch_idx, (cnn_features, semantic_features, gt_cap, _) in enumerate(test_dataloader):
        print(f'Generating {batch_idx+1}/{test_dataloader_len}', end='\r')
        cnn_features = cnn_features.to(DEVICE)
        semantic_features = semantic_features.to(DEVICE)
        gt_cap = gt_cap.to(DEVICE)
        bos_cap = gt_cap[:, 0:1]  # contains only BOS tag

        out = model(bos_cap, cnn_features, semantic_features, mode=mode)  # (BATCH_SIZE, timestep-1, vocab_size)

        for i in range(len(out)):
            if mode == 'argmax':
                out_cap = torch.argmax(out[i], dim=1).to(DEVICE).long()
            else:
                out_cap = softmax_func(out[i])
                out_cap = torch.multinomial(out, 1).squeeze(dim=1).to(DEVICE).long()

            out_cap = idx_to_annotation(out_cap.tolist(), test_dataset.idx_to_word)
            grount_truth = idx_to_annotation(gt_cap[i].tolist(), test_dataset.idx_to_word)

            out_cap = format_result(out_cap)
            grount_truth = format_result(grount_truth[1:])
            result['predictions'][str(batch_idx * batch_size + i + 1300)] = [out_cap]

    with open(out_path, 'w') as output:
        json.dump(result, output)
    print(f'\nGenerated caption to {out_path}')

except Exception:
    traceback.print_exc()

# python savc_generate_caption.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --dataset-dir "D:/ML Dataset/MSVD/features" --batch-size 64 --model-path "./checkpoints/savc/1643719115_epoch149_2.704_4.093_vgg_shufflenetv2.pth" --mode argmax --out-path "./payload.json"