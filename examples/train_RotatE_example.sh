#!/bin/bash

# Train RotatE on WN18RR with 500*500 relation and 500*1 entity.
python run.py \
            --dataset WN18RR \
            --model RotatE \
            --rank 500 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 4000 \
            --neg_sample_size 300 \
            --init_size 0.001 \
            --learning_rate_entity 0.1 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg