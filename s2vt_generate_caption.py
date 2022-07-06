import traceback
import argparse
import torch
import time
import json
from datetime import datetime
from torch.utils.data import DataLoader
from s2vt.dataset import PreprocessedMSVDDataset
from s2vt.model import S2VT
from utils import idx_to_annotation
from constant import *
from helper import *

parser = argparse.ArgumentParser(description='Predict using S2VT Model')
parser.add_argument('--annotation-path', help='File path to annotation', required=True)
parser.add_argument('--test-data-dir', help='Directory path to test data', required=True)
parser.add_argument('--out-path', help='Output filepath', required=True)
parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
parser.add_argument('--model-path', help='Load pretrained model', required=True)

args = parser.parse_args()
annotation_path = args.annotation_path
test_data_dir = args.test_data_dir
out_path = args.out_path
timestep = args.timestep
batch_size = args.batch_size
model_path = args.model_path

# show test config
print('\n######### TEST CONFIGURATION #########')
print('Annotation file:', annotation_path)
print('Output:', out_path)
print('Test directory:', test_data_dir)
print('Timestep:', timestep)
print('Pretrained model path:', model_path)
print('Batch size:', batch_size)

# prepare train and validation dataset
test_dataset = PreprocessedMSVDDataset(
    test_data_dir,
    annotation_path,
    timestep=timestep,
)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# create and prepare model
model = S2VT(
    word_to_idx=test_dataset.word_to_idx,
    vocab_size=test_dataset.vocab_size,
    timestep=timestep,
    lstm_hidden_size=500,
    drop_out_rate=0.3,
).to(DEVICE)

print(f'\nLoading pretrained model in {model_path}\n')
model.load_state_dict(torch.load(model_path))

model.eval()

try:
    uid = int(time.time())

    test_dataloader_len = len(test_dataloader)
    result = {'email': 'iam.steve.immanuel@gmail.com', 'predictions': {}}

    for batch_idx, (X, (y, y_mask)) in enumerate(test_dataloader):
        print(f'Generating {batch_idx+1}/{test_dataloader_len}', end='\r')
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y_mask = y_mask.to(DEVICE)

        out = model(X)
        out = torch.argmax(out, dim=2).to(DEVICE).long()

        for i in range(len(out)):
            caption = idx_to_annotation(out[i].tolist(), test_dataset.idx_to_word)
            grount_truth = idx_to_annotation(y[i].tolist(), test_dataset.idx_to_word)

            caption = format_result(caption)
            grount_truth = format_result(grount_truth[1:])
            result['predictions'][str(batch_idx * batch_size + i + 1300)] = [caption]

    with open(out_path, 'w') as output:
        json.dump(result, output)
    print(f'\nGenerated caption to {out_path}')

except Exception:
    traceback.print_exc()

# python generate_caption.py --annotation-path "D:/ML Dataset/MSVD/annotations.txt" --test-data-dir "C:/MSVD_extracted/out/testing" --batch-size 10 --model-path "./checkpoints/1642771058_epoch099_0.198_0.800.pth" --out-path "./payload.json"