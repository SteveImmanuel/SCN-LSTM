# Automatic Video Captioning using Deep Learning
This repo contains video captioning using deep learning implementation of :
- S2VT model as described in <a href=https://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf>this paper</a>.
- SCN-LSTM with Scheduled Sampling as described in <a href=https://arxiv.org/pdf/1909.00121.pdf>this paper</a>

## Setup
To install all depedencies, I suggest using <a href=https://docs.conda.io/en/latest/miniconda.html>miniconda</a> to configure virtual environment easily. After installing miniconda, create the environment using `env` file according to your operating system by typing:
```
//on windows
conda env create -f env-windows.yml

//on linux
conda env create -f env-linux.yml
```

## Preparing Dataset
### S2VT
To be added
### SCN-LSTM
Download the extracted MSVD dataset from <a href=https://drive.google.com/file/d/1LV5HMmbllnomHlZ2CQ-7QGE1Al9F0Qwi/view>here</a>. It contains CNN features extracted from ECO and ResNext architecture, and also semantics features as described in the original paper. Extract the files `msvd_resnext_eco.npy`, `msvd_semantic_tag_e1000.npy`, `msvd_tag_gt_4_msvd.npy` and put all of them into a single directory.

## Training
### S2VT
To be added
### SCN-LSTM
To train from scratch, use the following command:
```
python savc_train.py --annotation-path <PATH_TO_ANNOTATION> --dataset-dir <PATH_TO_DATASET> --batch-size 64 --epoch 150 --learning-rate 5e-4 --ckpt-interval 10 --model-cnn-2d "resnext" --model-cnn-3d "eco"  --mode sample --timestep 30
```

## Generate Captions
```
python savc_generate_caption.py --annotation-path <PATH_TO_ANNOTATION> --dataset-dir <PATH_TO_DATASET> --batch-size 64 --model-path <PRETRAINED_MODEL_PATH> --mode argmax --out-path "./payload.json"
```

## Summary
### S2VT
```
CIDEr: 0.6043070425344274
Bleu_4: 0.3966372294884941
Bleu_3: 0.5054434967645823
Bleu_2: 0.6120306365847461
Bleu_1: 0.7489382962776046
ROUGE_L: 0.6659791835245382
METEOR: 0.3006579474917490
```
### SCN-LSTM Result
```
CIDEr: 1.1695504045547098
Bleu_4: 0.6404037901180769
Bleu_3: 0.7334661777729088
Bleu_2: 0.8244963341487743
Bleu_1: 0.9204766536962741
ROUGE_L: 0.7941562007691393
METEOR: 0.4155076768981118
```

## References
- https://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf
- https://arxiv.org/pdf/1909.00121.pdf
- https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning
- https://github.com/mzolfaghari/ECO-pytorch
- https://github.com/SteveImmanuel/Efficient-3DCNNs
- https://github.com/YiyongHuang/S2VT