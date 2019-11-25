#!/bin/bash

python3 trainer.py \
--name train01 \
--logger Experiment --logger_api_key 1zB8P6u9ztqAuzy88PWhpbaIU --logger_project_name piv-flownet --logger_workspace flow-diagnostics-itb \
--training_dataset_root ../piv_datasets/cai2018/ztest_json \
--validation_dataset_root', '../piv_datasets/cai2018/ztest_json