python3 train_for_all_1.py --config configs/few-shot-setting/Template-BBDM-tfbind8-avg-full-data.yaml --save_top 

python3 train_and_test_for_all.py --config configs/few-shot-setting/Template-BBDM-tfbind8-avg-full-data.yaml \
&python3 train_and_test_for_all.py --config configs/few-shot-setting/Template-BBDM-tfbind8-rand-full-data.yaml \
&python3 train_and_test_for_all_1.py --config configs/few-shot-setting/Template-BBDM-tfbind8-avg-full-data.yaml \
&python3 train_and_test_for_all_1.py --config configs/few-shot-setting/Template-BBDM-tfbind8-rand-full-data.yaml \
&python3 train_and_test_for_all_2.py --config configs/few-shot-setting/Template-BBDM-tfbind8-avg-full-data.yaml \
&python3 train_and_test_for_all_2.py --config configs/few-shot-setting/Template-BBDM-tfbind8-rand-full-data.yaml \
&python3 train_and_test_for_all_3.py --config configs/few-shot-setting/Template-BBDM-tfbind8-avg-full-data.yaml \
&python3 train_and_test_for_all_3.py --config configs/few-shot-setting/Template-BBDM-tfbind8-rand-full-data.yaml