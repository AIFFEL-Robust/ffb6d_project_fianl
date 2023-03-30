#!/bin/bash
n_gpu=1  # number of gpu to use
bst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar
#bst_mdl=train_log/ycb/best_checkpoint/FFB6D_prt.pth.tar
opt_l="O0"
#-checkpoint $trt_mdl
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu custom_train.py --gpus=$n_gpu -checkpoint $bst_mdl
