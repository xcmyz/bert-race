#!/bin/bash

# source /mnt/pan/admin/softwares-all-in-one/env.sh
# spack load gcc@7.5.0
# spack load py-torch
# spack load py-tqdm
# spack load py-torch-nvidia-apex
# spack load py-boto3
# spack load py-requests
# spack load py-tensorboardx

eval $(~/asc21/lhh/miniconda3/condabin/conda shell.bash activate asc21)
module load cudnn/7.6.4-CUDA10.1
module load proxy

export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=3 python3 test_race.py \
    --data_dir=./RACE \
    --bert_model=./large_models/ \
    --vocab_file=./bert-large-uncased-vocab.txt \
    --output_dir=result \
    --max_seq_length=512 \
    --do_lower_case \
    --eval_batch_size 32 \
    adam_320 adam_512 radam_320
