#!/bin/bash


# Train OrthogonalE on WN18RR with 500*1 entity, 500*500 relation, block size 2
python run.py \
    --dataset WN18RR \
    --model OrthogonalE \
    --entity_size_n 500 \
    --entity_size_m 1 \
    --block_size 2 \
    --optimizer Adagrad \
    --max_epochs 500 \
    --patience 15 \
    --valid 5 \
    --batch_size 4000 \
    --neg_sample_size 300 \
    --init_size 0.001 \
    --learning_rate_entity 0.2 \
    --learning_rate_relation 0.02 \
    --gamma 0.0 \
    --bias learn \
    --dtype double \
    --double_neg


# Train OrthogonalE on FB15k-237 with 1000*1 entity, 1000*1000 relation, block size 2
python run.py \
    --dataset FB237 \
    --model OrthogonalE \
    --entity_size_n 1000 \
    --entity_size_m 1 \
    --block_size 2 \
    --optimizer Adagrad \
    --max_epochs 2 \
    --patience 15 \
    --valid 1 \
    --batch_size 2000 \
    --neg_sample_size 300 \
    --init_size 0.001 \
    --learning_rate_entity 0.5 \
    --learning_rate_relation 0.05 \
    --gamma 0.0 \
    --bias learn \
    --dtype double \
    --double_neg


