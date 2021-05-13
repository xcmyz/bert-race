#!/bin/bash

source /mnt/pan/admin/softwares-all-in-one/env.sh
spack load gcc@7.5.0
spack load py-torch
spack load py-tqdm
spack load py-torch-nvidia-apex
spack load py-boto3
spack load py-requests
spack load py-tensorboardx

NODE_RANK=$1
MASTER_ADDR=$2
CORE_ID=60  # prefer starting from 60
NEW_MODEL=$3
USE_ADAM=$4
MODEL_NAME=$5
DATANAME=$6
MAX_SEQ=$7
BATCH_SIZE=$8

GPUS=0,1
NGPU_PER_NODE=2
NNODE=3
NODE_RANK=$NODE_RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=12345
CUDA_VISIBLE_DEVICES=$GPUS python3 launch.py \
    --nproc_per_node=$NGPU_PER_NODE \
    --nnodes=$NNODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --core_id=$CORE_ID \
    run_race.new.py \
    --NEW_MODEL=$NEW_MODEL \
    --USE_ADAM=$USE_ADAM \
    --data_dir=./ASC \
    --bert_model=./bert-large-uncased.tar.gz \
    --vocab_file=./bert-large-uncased-vocab.txt \
    --output_dir=$MODEL_NAME \
    --max_seq_length=$MAX_SEQ \
    --do_train \
    --do_lower_case \
    --train_batch_size=$BATCH_SIZE \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --num_train_epochs=2 \
    --gradient_accumulation_steps=1 \
    --fp16 \
    --loss_scale=128 \
    --dataname=$DATANAME
