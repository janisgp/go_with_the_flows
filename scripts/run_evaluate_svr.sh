
python evaluate_ae.py ./results/svr/ svr_model_warmup_36_4800 test 2500 2500 reconstruction --weights_type learned_weights --reps 1 --f1_threshold_lst 0.001 --cd --f1 --emd --unit_scale_evaluation
