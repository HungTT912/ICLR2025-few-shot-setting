python3 train_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-tfbind8-type_points_lowest.yaml --save_top \
&python3 train_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-tfbind8-type_points_random.yaml --save_top 

python3 train_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_25.yaml --save_top \
&python3 train_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_50.yaml --save_top 

python3 train_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_75.yaml --save_top

