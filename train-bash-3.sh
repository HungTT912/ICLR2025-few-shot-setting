python3 train_and_test_for_all.py --config configs/few-shot-setting/tune_3/Template-BBDM-ant-l6.0-d1.0.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/few-shot-setting/tune_3/Template-BBDM-ant-l6.0-d2.0.yaml --save_top \

python3 test_for_ant_tune_3.py --config configs/few-shot-setting/tune_3/Template-BBDM-ant-l6.0-d1.0yaml \
&python3 test_for_ant_tune_3.py --config configs/few-shot-setting/tune_3/Template-BBDM-ant-l6.0-2.0.yaml \
&python3 train_and_test_for_all.py --config configs/few-shot-setting/tune_3/Template-BBDM-ant-l6.0-d0.5.yaml --save_top \

python3 test_for_ant_tune_3.py --config configs/few-shot-setting/tune_3/Template-BBDM-ant-l6.0-d0.5yaml