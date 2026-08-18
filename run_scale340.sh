#!/bin/bash
# Scale study, point 2 of ladder (137M done -> 340M -> 760M).
# 340M: d_model 512, 12 layers, 16 experts top-2, d_expert 1024. adam8bit for 8GB.
# Arm A (baseline) vs Arm B (locality lambda=0.05) at the SAME 200M-token budget as
# the 137M runs, so the A-vs-B PPL gap is directly comparable across scale.
set -e
cd /mnt/c/Users/shrin/Desktop/AI/sticky-moe
COMMON="--d-model 512 --n-layers 12 --n-experts 16 --d-expert 1024 --bsz 4 --adam8bit --tokens 200e6 --seed 1"
echo "=== $(date) START scale340 arm A ==="
python3 train.py --arm A --run-name scale340-a $COMMON
echo "=== $(date) START scale340 arm B lambda=0.05 ==="
python3 train.py --arm B --lambda-loc 0.05 --run-name scale340-b05 $COMMON
touch runs/SCALE340.FINISHED
echo "=== $(date) DONE scale340 ==="
