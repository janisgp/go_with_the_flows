#!/bin/bash -l
#!/bin/bash -l
python train_ae.py ./results/airplane_gen_model/config.yaml airplane_gen_mode 800 0.000256 --weights_type learned_weights --warmup_epoch 5 --distributed
python train_ae.py ./results/airplane_gen_model/config.yaml airplane_gen_model 1200 0.000064 --resume --weights_type learned_weights --distributed
python train_ae.py ./results/airplane_gen_model/config.yaml airplane_gen_model 1400 0.000016 --resume --weights_type learned_weights --distributed
python train_ae.py ./results/airplane_gen_model/config.yaml airplane_gen_model 1450 0.000004 --resume --weights_type learned_weights --distributed