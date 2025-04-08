# WST: Wavelet-Based Multi-scale Tuning for Visual Transfer Learning

This is the implementation of "WST: Wavelet-Based Multi-scale Tuning for Visual Transfer
Learning" .




<h3>Abstract</h3>
    
We present a novel efficient tuning method Wavelet-based multi-Scale Tuning (WST). Due to the intricate nature of downstream tasks requiring a finer level of details, 
we devote to incorporating small-scale features. We introduce an additional parallel small-scale patch embedding branch with a smaller patch size 
alongside the pre-trained backbone. This enables the model to concentrate on a smaller patch receptive field, thereby learning fine-grained 
features. To balance the computation and performance, 
we employ wavelet transform for lossless down-sampling of the token sequence. 
This facilitates matching token sequence sizes and enables the efficient fusion of multi-scale features.
Our WST achieves promising performance with small parameter count.





## Environment


- python 3.8
- pytorch >= 1.7
- torchvision >= 0.8
- timm 0.5.4

## Data Preparation

### 1. Visual Task Adaptation Benchmark (VTAB-1K)

Please refer to `VPT` or `NOAH` to construct VTAB-1K data benchmark.

`References:` 

`VPT: Jia, M.; Tang, L.; Chen, B.-C.; Cardie, C.; Belongie, S.; Hariharan, B.; and Lim, S.-N. 2022. Visual prompt tuning. In European Conference on Computer Vision, 709-727. Springer.`

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

The file structure should look like:
```
$ROOT/vtab-1k
|-- cifar
|-- caltech101
...
|-- diabetic_retinopathy
...
|-- svhn
```

## Pre-trained Models
Please download the weights of ViT-B/16 pre-trained on ImageNet-21K.

For Swin-B, the pre-trained weights will be automatically download to cache directory when you run training scripts.
## Usage

### Fine-tuning ViT-B/16 on VTAB-1K

```
bash train_scripts/vit/vtab/$DATASET_NAME/train_wst.sh
```
- Replace `DATASET_NAME` with the name you want for your dataset.
- Update the `data_dir` and `load_path` variables in the script to your specified `vtab-1k` path and `ViT-B pre-trained model` path.
- The hyperparameters such as 'scale', 'lr', 'drop_path' are needed to be tuned.
- The example files have been uploaded (caltech101 and dtd).
### Acknowledgement
The code is built upon `timm`, `WaveVIT`, `VPT`, `NOAH` and `DTL`.

`References:` 

`WaveViT: Yao, T.; Pan, Y.; Li, Y.; Ngo, C.-W.; and Mei, T. 2022. Wavevit: Unifying wavelet and transformers for visual representation learning. In European Conference on Computer Vision, 328-345. Springer.`

`VPT: Jia, M.; Tang, L.; Chen, B.-C.; Cardie, C.; Belongie, S.; Hariharan, B.; and Lim, S.-N. 2022. Visual prompt tuning. In European Conference on Computer Vision, 709-727. Springer.`

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

`DTL: Fu, M.; Zhu, K.; and Wu, J. 2024. Dtl: Disentangled transfer learning for visual recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 12082-12090.`
