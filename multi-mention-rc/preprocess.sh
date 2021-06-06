#!/bin/bash

if [ "$1" = "quasart" ]; then
    train_file="data/$1/$1-train0.json,data/$1/$1-train1.json,data/$1/$1-train2.json,data/$1/$1-train3.json"
elif [ "$1" = "webquestions" ]
then
    train_file="data/$1/$1-train0.json,data/$1/$1-train1.json"
else
    train_file="data/$1/$1-train.json"
fi

cmd="python run_data_preprocess.py \
        --log_dir ./data/$1 \
        --train_file ${train_file} \
        --test_file data/$1/$1-test.json \
        --eval_file data/$1/$1-dev.json \
        --verbose"

eval ${cmd}
