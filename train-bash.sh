python3 train_for_all_1.py --config configs/few-shot-setting/Template-BBDM-tfbind10-rand.yaml --save_top \
&python3 train_for_all_1.py --config configs/few-shot-setting/Template-BBDM-dkitty-rand.yaml --save_top

python3 train_and_test_for_all.py --config configs/few-shot-setting/Template-BBDM-tfbind10-rand.yaml \
&python3 train_and_test_for_all_1.py --config configs/few-shot-setting/Template-BBDM-tfbind10-rand.yaml \
&python3 train_and_test_for_all_2.py --config configs/few-shot-setting/Template-BBDM-tfbind10-rand.yaml \
&python3 train_and_test_for_all_3.py --config configs/few-shot-setting/Template-BBDM-tfbind10-rand.yaml

python3 train_and_test_for_all.py --config configs/few-shot-setting/Template-BBDM-dkitty-rand.yaml \
&python3 train_and_test_for_all_1.py --config configs/few-shot-setting/Template-BBDM-dkitty-rand.yaml \
&python3 train_and_test_for_all_2.py --config configs/few-shot-setting/Template-BBDM-dkitty-rand.yaml \
&python3 train_and_test_for_all_3.py --config configs/few-shot-setting/Template-BBDM-dkitty-rand.yaml