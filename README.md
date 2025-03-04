# WST: Wavelet-Based Multi-scale Tuning for Visual Transfer Learning

This is the implementation of "WST: Wavelet-Based Multi-scale Tuning for Visual Transfer
Learning" .

## Data Preparation

### 1. Visual Task Adaptation Benchmark (VTAB-1K)

Please refer to `VPT` or `NOAH` to construct VTAB-1K data benchmark.

`References:` 

`VPT: Jia, M.; Tang, L.; Chen, B.-C.; Cardie, C.; Belongie, S.; Hariharan, B.; and Lim, S.-N. 2022. Visual prompt tuning. In European Conference on Computer Vision, 709–727. Springer.`

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

### 2. Few-Shot Learning
Few-shot learning data benchmark consists of five fine-grained datasets: (`fgvc-aircraft, food101, oxford-flowers102, oxford-pets, standford-cars`).
- Images
   
   Please refer to `NOAH` or `DTL` to construct few-shot learning data benchmark.

`References:` 

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

`DTL: Fu, M.; Zhu, K.; and Wu, J. 2024. Dtl: Disentangled transfer learning for visual recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 12082–12090.`

### 3. Domain Generalization
Domain Generalization consists of one source dataset: (`ImageNet`) and four target datasets (`ImageNet-A, ImageNet-R, ImageNetV2, ImageNet-Sketch`): 
- Images

  Please refer to `CoOP` and `NOAH` to download for (`ImageNet, ImageNet-A, ImageNet-R, ImageNetV2, ImageNet-Sketch`) datasets.

`References:` 

`CoOP: Zhou, K.; Yang, J.; Loy, C. C.; and Liu, Z. 2022. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9): 2337–2348.`

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

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

### Fine-tuning Swin-B on VTAB-1K
```
bash train_scripts/swin/vtab/$DATASET_NAME/train_wst.sh
```

- Replace `DATASET_NAME` with the name you want for your dataset.
- Update the `data_dir` variable in the script to your specified `vtab-1k` path.

### Fine-tuning ViT-B/16 on Few-shot Learning

```
bash train_scripts/vit/few_shot/$DATASET_NAME/train_wst_shot_$SHOT.sh
```

- Replace `DATASET_NAME` with the name you want for your dataset.
- Replace `SHOT` with the training shot you want.
- Update the `data_dir` and `load_path` variables in the script to your specified `FGFS` path and `ViT-B pre-trained model` path.


### Fine-tuning ViT-B/16 on Domain Generalization
```
bash train_scripts/vit/domain_generalization/$DATASET_NAME/train_wst.sh
```

- Replace `DATASET_NAME` with the name you want for your dataset.
- Update the `data_dir` and `load_path` variables in the script to your specified `DG` path and `ViT-B pre-trained model` path.


### Acknowledgement
The code is built upon `timm`, `WaveViT`, `VPT`, `NOAH` and `DTL`.

`References:` 

`WaveViT: Yao, T.; Pan, Y.; Li, Y.; Ngo, C.-W.; and Mei, T. 2022. Wavevit: Unifying wavelet and transformers for visual representation learning. In European Conference on Computer Vision, 328–345. Springer.`

`VPT: Jia, M.; Tang, L.; Chen, B.-C.; Cardie, C.; Belongie, S.; Hariharan, B.; and Lim, S.-N. 2022. Visual prompt tuning. In European Conference on Computer Vision, 709–727. Springer.`

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

`DTL: Fu, M.; Zhu, K.; and Wu, J. 2024. Dtl: Disentangled transfer learning for visual recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 12082–12090.`
