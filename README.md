# Automatic Video Captioning using Deep Learning
This repo contains video captioning using deep learning implementation using PyTorch of :
- S2VT model as described in <a href=https://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf>this paper</a>.
- SCN-LSTM with Scheduled Sampling as described in <a href=https://arxiv.org/pdf/1909.00121.pdf>this paper</a>

Detailed explanation can be found <a href="https://docs.google.com/presentation/d/1LEnscxa_wEj83AUvil5NVvW76Ap1KkEn1PuIY6Dbg8g/edit?usp=sharing">here</a>.
## Setup
To install all depedencies, I suggest using <a href=https://docs.conda.io/en/latest/miniconda.html>miniconda</a> to configure virtual environment easily. After installing miniconda, create the environment using `env` file according to your operating system by typing:
```
//on windows
conda env create -f env-windows.yml

//on linux
conda env create -f env-linux.yml
```

## Preparing Dataset

Download the annotation file from <a href="https://drive.google.com/file/d/1NRCPSSmdKH0djFczJ1C12uVPIwZHkj_6/view?usp=sharing">here</a>.

### S2VT
Download the extracted MSVD dataset from <a href=https://www.kaggle.com/steveandreasimmanuel/msvd-extracted>here</a>. It contains 3 directory which are `train`, `testing`, and `validation`. Each folder contains `.npy` files, each contains CNN features extracted using VGG for a video.

### SCN-LSTM
Download the extracted MSVD dataset from <a href=https://drive.google.com/file/d/1LV5HMmbllnomHlZ2CQ-7QGE1Al9F0Qwi/view>here</a>. It contains CNN features extracted from ECO and ResNext architecture, and also semantics features as described in the original paper. Extract the files `msvd_resnext_eco.npy`, `msvd_semantic_tag_e1000.npy`, `msvd_tag_gt_4_msvd.npy` and put all of them into a single directory.

## Training
### S2VT
To train from scratch, use the following command:
```
python train.py \
--annotation-path <PATH_TO_ANNOTATION> \
--train-data-dir <PATH_TO_TRAIN_DIR> \
--val-data-dir <PATH_TO_VAL_DIR> \
--batch-size 10 --epoch 800 --learning-rate 1e-4 --ckpt-interval 10
```
### SCN-LSTM
To train from scratch, use the following command:
```
python savc_train.py \
--annotation-path <PATH_TO_ANNOTATION>\
 --dataset-dir <PATH_TO_DATASET> \
 --batch-size 64 --epoch 150 --learning-rate 5e-4 --ckpt-interval 10 \
 --model-cnn-2d "resnext" --model-cnn-3d "eco" \
 --mode sample --timestep 30
```

## Generate Captions

### S2VT
```
python generate_caption.py \
--annotation-path <PATH_TO_ANNOTATION> \
--test-data-dir <PATH_TO_TEST_DIR> \
--model-path <PRETRAINED_MODEL_PATH> \
--batch-size 10 \
--out-path <OUT_PATH>
```

### SCN-LSTM
```
python savc_generate_caption.py \
--annotation-path <PATH_TO_ANNOTATION> \
--dataset-dir <PATH_TO_DATASET> \
--model-path <PRETRAINED_MODEL_PATH> \
--batch-size 64 --mode argmax --out-path <OUT_PATH>
```

## Evaluation
To get the score metrics on Blue, CiDEr,  ROUGE_L, and METEOR, generate the captions and use <a href=https://github.com/SteveImmanuel/caption-eval>this</a> repository. 

## Performance on MSVD
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
### SCN-LSTM
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
- Venugopalan, S., Rohrbach, M., Donahue, J., Mooney, R., Darrell, T., & Saenko, K. (2015). Sequence to sequence - Video to text. Proceedings of the IEEE International Conference on Computer Vision, 2015 Inter, 4534–4542. https://doi.org/10.1109/ICCV.2015.515
- Chen, H., Lin, K., Maye, A., Li, J., & Hu, X. (2020). A Semantics-Assisted Video Captioning Model Trained With Scheduled Sampling. Frontiers in Robotics and AI, 7, 1–11. https://doi.org/10.3389/frobt.2020.475767
- https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning
- https://github.com/mzolfaghari/ECO-pytorch
- https://github.com/okankop/Efficient-3DCNNs
- https://github.com/YiyongHuang/S2VT