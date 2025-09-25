#!/bin/bash

python train.py \
    --dataset convex_hull \
    --model convex_hull_glgmlp \
    --dataroot datasets \
    --n 5 \
    --num_samples 256 \
    --max_steps 512 \
    --batch_size 128 \
    --hidden_features 32 \
    --num_layers 4 \
    --subspace_type "Q" \
    --num_workers 4 \
    --prefetch_factor 2 

