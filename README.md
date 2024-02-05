# OrthogonalE

This is the PyTorch implementation of the [OrthogonalE](https://arxiv.org/abs/2401.05967) [1] model for knowledge graph embedding (KGE). 
This project is based on [AttH](https://github.com/HazyResearch/KGEmb) [2] and [3H-TH](http://arxiv.org/abs/2305.13015) [3]. Thanks for their contributions.


## Models

* TransE  [4]
* RotatE  [5]
* QuatE   [6]
* OrthogonalE(2*2) [1]
* OrthogonalE(3*3) [1]


## Initialization

1. environment (we need torch, numpy, tqdm, geoopt):

```bash
conda create --name OrthogonalE_env
source activate OrthogonalE_env
pip install -r requirements.txt
```

2. set environment variables.

we should set envirment variables for experiment.

```bash
KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:$PYTHONPATH"
export LOG_DIR="$KGHOME/logs"
export DATA_PATH="$KGHOME/data"
```
Then we can activate our environment:

```bash
source activate OrthogonalE_env
```

## Data

I have uploaded all the data that we need to use in the data document.
While the FB15K-237 is large, so we need to unzip the big dataset as following:

```bash
cd data/FB237
unzip to_skip.pickle.zip
```

## usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script. And we can use the file "examples", "train_OrthogonalE_example.sh" and "train_RotatE.sh" means the examples for OrthogonalE and RotatE, respectively.

```bash
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,ComplEx,RotatE,QuatE,OrthogonalE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad, Adam, SparseAdam, RiemannianSGD, RiemannianAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK]
              [--entity_size_n ENTITY_SIZE_N]
              [--entity_size_m ENTITY_SIZE_M]
              [--block_size BLOCK_SIZE]
              [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE]
              [--learning_rate_entity LEARNING_RATE_ENTITY]
              [--learning_rate_relation LEARNING_RATE_RELATION]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN18RR,FB237}
                        Knowledge Graph dataset
  --model {TransE,RotatE,QuatE,RotH,ThreeE_TE,TwoE_TE,TH,ThreeH,ThreeH_TH,ThreeE_TE_ThreeH_TH, TwoE_TE_TwoH_TH}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --entity_size_n       the size of entity matrix for n
  --entity_size_m       the size of entity matrix for m
  --block_size          the size of block-diagonal matrix
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate_entity
                        Learning rate for the entity in OrthogonalE train process
  --learning_rate_relation
                        Learning rate for the relation in OrthogonalE train process
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  
```
For example:  Train OrthogonalE on WN18RR with 500_1 entity, 500_500 relation, block size 2

```bash
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
    --neg_sample_size 200 \
    --init_size 0.001 \
    --learning_rate_entity 0.2 \
    --learning_rate_relation 0.02 \
    --gamma 0.0 \
    --bias learn \
    --dtype double \
    --double_neg
```

## Citation

If you want to cite this paper or want to use this code, please cite the following paper:

```
@article{zhu2024block,
  title={Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding},
  author={Zhu, Yihua and Shimodaira, Hidetoshi},
  journal={arXiv preprint arXiv:2401.05967},
  year={2024}
}
```

## Reference

[1] Zhu, Yihua, and Hidetoshi Shimodaira. "Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding." arXiv preprint arXiv:2401.05967 (2024).

[2] Chami I, Wolf A, Juan D C, et al. Low-dimensional hyperbolic knowledge graph embeddings[J]. arXiv preprint arXiv:2005.00545, 2020.

[3] Zhu Y, Shimodaira H. 3D Rotation and Translation for Hyperbolic Knowledge Graph Embedding[J]. arXiv preprint arXiv:2305.13015, 2023.

[4] Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.

[5] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." International Conference on Learning Representations. 2019.

[6] Zhang S, Tay Y, Yao L, et al. Quaternion knowledge graph embeddings[J]. Advances in neural information processing systems, 2019, 32.
