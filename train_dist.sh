#!/bin/bash

source /etc/profile
source /mnt/work/env-tf/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export PS_LIST=52.247.207.135:5999
export WORKER_LIST=52.183.86.201:5999,52.247.196.123:5999
export SECONDS=60480000
export LOGSUFFIX=BIGLSTM
export myip="$(dig +short myip.opendns.com @resolver1.opendns.com)"

python3 dist_lm_train.py --logdir=./$LOGSUFFIX --num_gpus=$NUM_GPUS --datadir=../1-billion-word-language-modeling-benchmark/ \
         --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=$NUM_GPUS,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=64 \
         --ps_list=$PS_LIST --worker_list=$WORKER_LIST --my_ip=$myip
