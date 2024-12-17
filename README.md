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
- timm 0.5.4 (For Swin-B on VTAB-1K experiment, timm 0.6.5 is used.)

## Data Preparation

### 1. Visual Task Adaptation Benchmark (VTAB-1K)

Please refer to `VPT` or `NOAH` to construct VTAB-1K data benchmark.

`References:` 

`VPT: Jia, M.; Tang, L.; Chen, B.-C.; Cardie, C.; Belongie, S.; Hariharan, B.; and Lim, S.-N. 2022. Visual prompt tuning. In European Conference on Computer Vision, 709–727. Springer.`

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

### 2. Few-Shot Learning
Few-shot learning data benchmark consists of five fine-grained datasets: (`fgvc-aircraft, food101, oxford-flowers102, oxford-pets, standford-cars`).
  
- Images
   
   Please refer to `NOAH` or `DTL` to construct few-shot learning data benchmark.
  Images from the five datasets should be organized and consolidated into a folder named ***FGFS***.
- Train/Val/Test splits

   Please create a folder named ***few-shot_split*** and place `data/few-shot` in `NOAH` in this folder.

`References:` 

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`

`DTL: Fu, M.; Zhu, K.; and Wu, J. 2024. Dtl: Disentangled transfer learning for visual recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 12082–12090.`

The file structure should look like:
  ```
  $ROOT/FGFS
  ├── few-shot_split
  │   ├── fgvc-aircraft
  │   │   └── annotations
  │   │       ├── train_meta.list.num_shot_1.seed_0
  │   │       └── ...
  │   │    ...
  │   └── standford-cars
  │       └── annotations
  │           ├── train_meta.list.num_shot_1.seed_0
  │           └── ...
  ├── fgvc-aircraft
  │   ├── img1.jpeg
  │   ├── img2.jpeg
  │   └── ...
  │   ...
  └── standford-cars
      ├── img1.jpeg
      ├── img2.jpeg
      └── ...
  ```

### 3. Domain Generalization
Domain Generalization consists of one source dataset: (`ImageNet`) and four target datasets (`ImageNet-A, ImageNet-R, ImageNetV2, ImageNet-Sketch`): 
- Images

  Please refer to `CoOP` to download for (`ImageNet, ImageNet-A, ImageNet-R, ImageNetV2, ImageNet-Sketch`) datasets.
- Test splits

    Please refer to `NOAH` to copy `data/domain-generalization`. The directory ***annotations*** should be placed in a subdirectory for each dataset.
  
`References:` 

`CoOP: Zhou, K.; Yang, J.; Loy, C. C.; and Liu, Z. 2022. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9): 2337–2348.`

`NOAH: Zhang, Y.; Zhou, K.; and Liu, Z. 2024. Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence.`



The file structure should look like:
  ```
  $ROOT/DG
  ├── imagenet
  │   ├── annotations
  │   │       ├── train_meta.list.num_shot_1.seed_0
  │   │       └── ......
  │   │   
  │   └── images
  │ 
  ├── imagenet-adversarial
  │   ├── annotations
  │   │       ├── train_meta.list.num_shot_1.seed_0
  │   │       └── ....
  │   │   
  │   └── imagenet-a
  ├── imagenet-rendition
  │   ├── annotations
  │   │       ├── train_meta.list.num_shot_1.seed_0
  │   │       └── .....
  │   │   
  │   └── imagenet-r
  │── imagenet-sketch
  │   ├── annotations
  │   │       ├── train_meta.list.num_shot_1.seed_0
  │   │       └── ...
  │   │   
  │   └── images
  │── imagenetv2
       ├── annotations
       │       ├── train_meta.list.num_shot_1.seed_0
       │       └── ...
       │   
       └── imagenetv2-matched-frequency-format-val
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
