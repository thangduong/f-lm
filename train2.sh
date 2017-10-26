#!/bin/bash

source /etc/profile
source /mnt/work/env-tf/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

export SECONDS=60480000
export LOGSUFFIX=BIGLSTM
#train
python3 single_lm_train.py --logdir=./$LOGSUFFIX --num_gpus=$NUM_GPUS --datadir=../1-billion-word-language-modeling-benchmark/ \
         --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=$NUM_GPUS,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=64 \
#          > train_$LOGSUFFIX.log 2>&1

#eval
#python /home/okuchaiev/repos/f-lm/single_lm_train.py --logdir=/raid/okuchaiev/Workspace/LM/FGLSTM/$LOGSUFFIX --num_gpus=8 --datadir=/raid/okuchaiev/Data/LM1B/1-billion-word-language-modeling-benchmark-r13output/ --mode=eval_full --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=16 > eval_full_$LOGSUFFIX.log 2>&1
