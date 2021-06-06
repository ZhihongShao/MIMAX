#!/bin/bash

export CUDA_VISIBLE_DEVICES=

TASK=$1

# [mml, hard-em, hard-em-thres, mimax]
ALGO=$2

if [ "${ALGO}" = "hard-em" ]; then
  if [ "${TASK}" = "quasart" ]; then
    TAU=20000
  elif [ "${TASK}" = "webquestions" ]
  then
    TAU=30000
  else
    TAU=0
  fi
  ckpt_dir="${TASK}-${ALGO}-${TAU}"
elif [ "${ALGO}" = "mimax" ]
then
  if [ "${TASK}" = "quasart" ]; then
    mi_steps=10000
  elif [ "${TASK}" = "webquestions" ]
  then
    mi_steps=7000
  else
    mi_steps=-1
  fi
  ckpt_dir="${TASK}-${ALGO}-${mi_steps}"
else
  ckpt_dir="${TASK}-${ALGO}"
fi

cmd="python main.py \
      --should_continue \
      --continue_from_best_ckpt \
      --ckpt_dir checkpoints/${ckpt_dir} \
      --do_predict \
      --num_shards 1 \
      --loss_type ${ALGO} \
      --data_dir data/${TASK} \
      --predict_dev_file ${TASK}-dev-300-40-20.pkl \
      --pretrained_model_type_for_prior bart-base \
      --pretrained_model_type_for_posterior bert-base-uncased \
      --evaluate_during_training \
      --train_batch_size 20 \
      --eval_batch_size 300 \
      --verbose \
      --predict_test_file ${TASK}-test-300-40-20.pkl \
      --n_paragraphs 40"

eval ${cmd}
