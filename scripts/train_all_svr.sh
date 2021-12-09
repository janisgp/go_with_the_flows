#!/bin/bash -l
python train_svr.py ./configs/config_SVR.yaml all_svr_model 20 0.000256 --weights_type learned_weights --warmup_epoch 1  --distributed
python train_svr.py ./configs/config_SVR.yaml all_svr_model 30 0.000064 --weights_type learned_weights  --distributed --resume
python train_svr.py ./configs/config_SVR.yaml all_svr_model 35 0.000016 --weights_type learned_weights  --distributed --resume
python train_svr.py ./configs/config_SVR.yaml all_svr_model 36 0.000004 --weights_type learned_weights  --distributed --resume
