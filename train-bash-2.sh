python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8-s5000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8-s7500-l5.0-lr0.05-d0.25.yaml --save_top 

python3 test_tfbind8_tune_22.py --config configs/tune_22/Template-BBDM-tfbind8-s5000-l5.0-lr0.05-d0.25.yaml \
&python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8-s20000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 test_tfbind8_tune_22.py --config configs/tune_22/Template-BBDM-tfbind8-s7500-l5.0-lr0.05-d0.25.yaml 

