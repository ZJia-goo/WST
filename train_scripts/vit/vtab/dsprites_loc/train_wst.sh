CUDA_VISIBLE_DEVICES=0 python train_vit_vtab.py \
  --data_dir  /path/to/vtab-1k \
  --load_path /path/to/Vit-B_16.npz \
  --dataset dsprites_loc \
  --model vit_base_patch16_224_in21k \
  --batch_size 32 \
  --batch_size_test 256 \
  --epochs 100 \
  --warmup_epochs 10 \
  --r 2 \
  --scale 1 \
  --weight_decay 0.05 \
  --lr 0.003 \
  --drop_path 0.3 \
  --amp \
  --prefetcher