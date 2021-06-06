#!/bin/bash

export CUDA_VISIBLE_DEVICES=

# [mml, hard-em, hard-em-thres, mimax]
ALGO=$1

common_args="--subsample \
            --extreme_case \
            --do_train \
            --num_shards 1 \
            --predict_train_file preproc_wikisql_train.jsonl \
            --predict_train_table_file preproc_wikisql_train_tables.jsonl \
            --predict_train_db_file preproc_wikisql_train.db \
            --predict_dev_file preproc_wikisql_dev.jsonl \
            --predict_dev_table_file preproc_wikisql_dev_tables.jsonl \
            --predict_dev_db_file preproc_wikisql_dev.db \
            --predict_test_file preproc_wikisql_test.jsonl \
            --predict_test_table_file preproc_wikisql_test_tables.jsonl \
            --predict_test_db_file preproc_wikisql_test.db \
            --pretrained_model_type_for_posterior bert-base-uncased \
            --pretrained_model_type_for_prior bart-base \
            --evaluate_during_training \
            --plm_adaptive_lr \
            --train_batch_size 10 \
            --eval_batch_size 64 \
            --save_steps 1000"

if [ "${ALGO}" = "mimax" ]; then
  cmd="python main.py ${common_args} \
        --ckpt_dir checkpoints/${ALGO}-9000 \
        --use_prior \
        --loss_type ${ALGO} \
        --mi_steps 9000"
elif [ "${ALGO}" = "mml" ] || [ "${ALGO}" = "hard-em" ] || [ "${ALGO}" = "hard-em-thres" ]
then
  cmd="python main.py ${common_args} \
        --ckpt_dir checkpoints/${ALGO} \
        --loss_type ${ALGO}"
else
  echo "${ALGO} is not supported"
  exit 1
fi

eval ${cmd}
