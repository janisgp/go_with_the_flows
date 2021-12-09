#!/bin/bash -l
python train_ae.py ./configs/config_autoencoding.yaml ae_model 400 0.000256 --weights_type learned_weights --warmup_epoch 1 --distributed
python train_ae.py ./configs/config_autoencoding.yaml ae_model 800 0.000064 --resume --weights_type learned_weights --distributed
python train_ae.py ./configs/config_autoencoding.yaml ae_model 1000 0.000016 --resume --weights_type learned_weights --distributed
python train_ae.py ./configs/config_autoencoding.yaml ae_model 1050 0.000004 --resume --weights_type learned_weights --distributed
