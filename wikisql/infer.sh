#!/bin/bash

export CUDA_VISIBLE_DEVICES=

# [mml, hard-em, hard-em-thres, mimax]
ALGO=$1

common_args="--do_predict \
            --should_continue \
            --continue_from_best_ckpt \
            --num_shards 1 \
            --predict_test_file preproc_wikisql_test.jsonl \
            --predict_test_table_file preproc_wikisql_test_tables.jsonl \
            --predict_test_db_file preproc_wikisql_test.db \
            --pretrained_model_type_for_posterior bert-base-uncased \
            --pretrained_model_type_for_prior bart-base \
            --train_batch_size 10 \
            --eval_batch_size 64"

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
