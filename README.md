# BERT for RACE dataset

## Run

1. multi-worker, multi-GPU:
    - `bash 2_run_multiworker.sh 0 <addr> 0 1 <model name> <dataset name> 320 <batch size on single GPU>`
2. evaluation:
    - `bash eval.sh`
