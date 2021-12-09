#!/bin/bash -l
python train_ae.py ./configs/config_generative_modeling_car.yaml car_gen_model 1000 0.000256 --weights_type learned_weights --warmup_epoch 5 --distributed
python train_ae.py ./configs/config_generative_modeling_car.yaml car_gen_model 1500 0.000064 --resume --weights_type learned_weights --distributed
python train_ae.py ./configs/config_generative_modeling_car.yaml car_gen_model 1750 0.000016 --resume --weights_type learned_weights --distributed
python train_ae.py ./configs/config_generative_modeling_car.yaml car_gen_model 1800 0.000004 --resume --weights_type learned_weights --distributed
