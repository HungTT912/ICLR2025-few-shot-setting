python3 train_and_test_for_all.py --config configs/few-shot-setting/tune_6/Template-BBDM-tfbind8-highest-15000.yaml --save_top \
&python3 test_for_all.py --config configs/few-shot-setting/tune_6/Template-BBDM-ant-rand-type-2.yaml --save_top

python3 train_and_test_for_all.py --config configs/few-shot-setting/tune_6/Template-BBDM-tfbind10-rand.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/few-shot-setting/tune_6/Template-BBDM-dkitty.yaml --save_top