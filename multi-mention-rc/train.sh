#!/bin/bash

export CUDA_VISIBLE_DEVICES=

TASK=$1

# [mml, hard-em, hard-em-thres, mimax]
ALGO=$2

if [ "${TASK}" = "quasart" ]; then
  train_file="${TASK}-train0-300-40-20.pkl,${TASK}-train1-300-40-20.pkl,${TASK}-train2-300-40-20.pkl,${TASK}-train3-300-40-20.pkl"
elif [ "${TASK}" = "webquestions" ]
then
  train_file="${TASK}-train0-300-40-20.pkl,${TASK}-train1-300-40-20.pkl"
else
  train_file="${TASK}-train-300-40-20.pkl"
fi

if [ "${ALGO}" = "mml" ]; then
  cmd="python main.py \
        --do_train \
        --num_shards 1 \
        --loss_type ${ALGO} \
        --data_dir data/${TASK} \
        --ckpt_dir checkpoints/${TASK}-${ALGO} \
        --summary_dir runs/${TASK}-${ALGO} \
        --predict_train_file ${train_file} \
        --predict_dev_file ${TASK}-dev-300-40-20.pkl \
        --pretrained_model_type_for_posterior bert-base-uncased \
        --evaluate_during_training \
        --train_batch_size 20 \
        --save_total_limit 3 \
        --eval_batch_size 300 \
        --ema_decay 0 \
        --mi_steps -1 \
        --save_steps 1000 \
        --verbose"
elif [ "${ALGO}" = "hard-em" ]
then
  if [ "${TASK}" = "quasart" ]; then
    TAU=20000
  elif [ "${TASK}" = "webquestions" ]
  then
    TAU=30000
  else
    TAU=0
  fi
  cmd="python main.py \
        --do_train \
        --num_shards 1 \
        --loss_type MIMAX \
        --data_dir data/${TASK} \
        --ckpt_dir checkpoints/${TASK}-${ALGO}-${TAU} \
        --summary_dir runs/${TASK}-${ALGO}-${TAU} \
        --predict_train_file ${train_file} \
        --predict_dev_file ${TASK}-dev-300-40-20.pkl \
        --pretrained_model_type_for_posterior bert-base-uncased \
        --evaluate_during_training \
        --train_batch_size 20 \
        --save_total_limit 3 \
        --eval_batch_size 300 \
        --ema_decay 0 \
        --mi_steps -1 \
        --save_steps 1000 \
        --verbose \
        --tau ${TAU}"
elif [ "${ALGO}" = "hard-em-thres" ]
then
  cmd="python main.py \
        --do_train \
        --num_shards 1 \
        --loss_type ${ALGO} \
        --data_dir data/${TASK} \
        --ckpt_dir checkpoints/${TASK}-${ALGO} \
        --summary_dir runs/${TASK}-${ALGO} \
        --predict_train_file ${train_file} \
        --predict_dev_file ${TASK}-dev-300-40-20.pkl \
        --pretrained_model_type_for_posterior bert-base-uncased \
        --evaluate_during_training \
        --train_batch_size 20 \
        --save_total_limit 3 \
        --eval_batch_size 300 \
        --ema_decay 0 \
        --mi_steps -1 \
        --save_steps 1000 \
        --verbose"
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
  cmd="python main.py \
        --use_prior \
        --do_train \
        --num_shards 1 \
        --loss_type ${ALGO} \
        --data_dir data/${TASK} \
        --ckpt_dir checkpoints/${TASK}-${ALGO}-${mi_steps} \
        --summary_dir runs/${TASK}-${ALGO}-${mi_steps} \
        --predict_train_file ${train_file} \
        --predict_dev_file ${TASK}-dev-300-40-20.pkl \
        --pretrained_model_type_for_prior bart-base \
        --pretrained_model_type_for_posterior bert-base-uncased \
        --evaluate_during_training \
        --train_batch_size 20 \
        --save_total_limit 3 \
        --eval_batch_size 300 \
        --mi_steps ${mi_steps} \
        --save_steps 1000 \
        --verbose"
else
  echo "${ALGO} is not supported"
  exit 1
fi

eval ${cmd}
