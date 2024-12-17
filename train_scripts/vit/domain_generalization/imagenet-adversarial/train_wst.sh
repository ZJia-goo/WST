CUDA_VISIBLE_DEVICES=0 python train_vit_domain_genralization.py \
  --data_dir  /path/to/DG \
  --load_path /path/to/Vit-B_16.npz \
  --source_dataset imagenet \
  --target_dataset imagenet-adversarial \
  --model vit_base_patch16_224_in21k \
  --batch_size 32 \
  --batch_size_test 256 \
  --epochs 100 \
  --warmup_epochs 10 \
  --r 2 \
  --scale 1.0 \
  --weight_decay 0.05 \
  --lr 0.00008 \
  --drop_path 0.0 \
  --amp \
  --prefetcher

