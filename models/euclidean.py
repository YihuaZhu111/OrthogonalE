"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

import geoopt


from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "QuatE", "OrthogonalE"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size, args.block_size, args.entity_size_m, args.entity_size_n)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.name = "TransE"
        self.sim = "dist"
    
    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.relation(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class QuatE(BaseE):          #  (QuatE)

    def __init__(self, args):
        super(QuatE, self).__init__(args)
        self.name = "QuatE"
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0     # (num_relations, rank)

        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""

        def _3d_rotation(relation, entity):    # entity:  (batch_size, rank)     relation: (batch_size, rank)
            relation_re, relation_i, relation_j, relation_k = torch.chunk(relation, 4, dim = -1)        # (batch_size, rank/4)
            entity_re, entity_i, entity_j, entity_k = torch.chunk(entity, 4, dim = -1)                  # (batch_size, rank/4)

            # normalize relation
            denominator_relation = torch.sqrt(relation_re ** 2 + relation_i ** 2 + relation_j ** 2 + relation_k **2)
            relation_re = relation_re / denominator_relation
            relation_i  = relation_i  / denominator_relation
            relation_j  = relation_j  / denominator_relation
            relation_k  = relation_k  / denominator_relation

            # do 3d rotation
            re = entity_re * relation_re - entity_i * relation_i - entity_j * relation_j - entity_k * relation_k
            i  = entity_re * relation_i  + entity_i * relation_re+ entity_j * relation_k - entity_k * relation_j
            j  = entity_re * relation_j  - entity_i * relation_k + entity_j * relation_re+ entity_k * relation_i
            k  = entity_re * relation_k  + entity_i * relation_j - entity_j * relation_i + entity_k * relation_re

            return torch.cat([re, i, j, k], dim = -1)             # (batch_size, rank)

        lhs_e = _3d_rotation(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class OrthogonalE(BaseE):             # orthogonal with Reimannian Stiefel Manifold optimization

    def __init__(self, args):
        super(OrthogonalE, self).__init__(args)
        
        self.name = "OrthogonalE"
        self.relation = geoopt.ManifoldParameter(
            torch.randn(self.sizes[1], int(self.entity_size_n / self.block_size), self.block_size, self.block_size),
            manifold=geoopt.EuclideanStiefel(canonical=False),)       

        self.entity = nn.Embedding(self.sizes[0], self.entity_size_n * self.entity_size_m) 
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.entity_size_n * self.entity_size_m), dtype=self.data_type)

        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor): 
        
        def _matrix_transformation(head, relation):    # entity:  (batch_size, entity_size_n * entity_size_m)  relation: (batch_size, entity_size_n/block_size, block_size, block_size)

            head_1 = head.view(int(head.size(0)), int(self.block_size), 1, int(self.entity_size_n / self.block_size), self.entity_size_m)    # (batch_size, block_size, 1, entity_size_n/block_size, entity_size_m) 

            relation_1 = relation.permute(0, 2, 3, 1)   # (batch_size, block_size, block_size, enity_size_n/block_size)
            relation_1 = relation_1.unsqueeze(-1)    #  (batch_size, block_size, block_size, enity_size_n/block_size, 1)
            relation_1 = relation_1.expand(-1, -1, -1, -1, self.entity_size_m)   # (batch_size, block_size, block_size, enity_size_n/block_size, entity_size_m) 
            relation_1 = relation_1.double()
            
            head_2 = torch.einsum('ijklh,ikmlh->ijmlh', relation_1, head_1) # (batch_size, block_size, 1, entity_size_n/block_size, entity_size_m)  
            head_2 = head_2.reshape(int(head_2.size(0)), -1)   # (batch_size, entity_size_n * entity_size_m)

            return head_2  # return transformed entity
        
        lhs_e = _matrix_transformation(self.entity(queries[:, 0]), self.relation[queries[:, 1]])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases



"""  
# gram_schmidt algorithm
class RotE(BaseE):    # entity matrix (30 * 10/20/30/40)   relation matrix (30 * 30)  (small block: 4*4)  rank = 30  orthogonal

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel = nn.Embedding(self.sizes[1], 6 * self.rank)
        self.rel.weight.data = 2 * torch.rand((self.sizes[1], 6 * self.rank), dtype=self.data_type) - 1.0     
       
        self.entity = nn.Embedding(self.sizes[0], 10 * self.rank)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], 10 * self.rank), dtype=self.data_type)
   
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
   
        def _matrix_transformation(entity, relation):    # entity:  (batch_size, 10 * rank)  relation: (batch_size, rank * 2)
  
            entity_1 = entity.view(int(entity.size(0)), 6, 1, int(entity.size(1)/60), 10)    # (batch_size, 2, 1, rank/2, 10)
            relation_1 = relation.view(int(relation.size(0)), 6, 6, int(self.rank/6))   # (batch_size, 2, 2, rank/2)
            
            def gram_schmidt(A):         # A: (Batch_size, 4, 4, rank/4)      # Parallel
                Q = torch.zeros_like(A)        # Q (Batch_size,4,4,rank/4)
                #for k in range(A.shape[3]):
                v = A    #    v : (Batch_size, 4, 4, rank/4)    现在v相当于A 
                for i in range(A.shape[1]):
                    n = torch.zeros_like(v)   # n : (Batch_size, 4, 4, rank/4)
                    # Begin iteration
                    n[:, i, :, :] = v[:, i, :, :]      # (Batch_size, 4, rank/4)
                    for j in range(i):
                    # Subtract the projection of A's current row onto Q's jth row
                        n_1 = torch.sum(v[:, i, :, :] * Q[:, j, :, :].clone(), dim = 1, keepdim = True)      # (Batch_size, 1, rank/4)
                        n[:, i, :, :] = n[:, i, :, :] - n_1 * Q[:, j, :, :].clone()    #(Batch_size, 4, rank/4)

                    # Normalize Q's ith row      n.norm(dim =2): (2,4,2)  ---- (2,1, 2)
                    # (Batch_size, 4, rank/4) / (Batch_size, 1, rank/4) = (Batch_size, 4, rank/4)
                    a = n[:, i, :, :] / n.norm(dim = 2)[:, i, :].unsqueeze(dim = 1)  
                    a = a.unsqueeze(1)                               #   (Batch_size,1,4,rank/4)
                    Q[:, i, :, :] = a[:, 0, :, :]
                return Q                   #(Batch_size, 4, 4, rank/4)
            
            relation_1 = gram_schmidt(relation_1)       # orthogonalization   (batch_size, 2, 2, rank/2)
            relation_1 = relation_1.unsqueeze(-1)    #  (batch_size, 2, 2, rank/2, 1)
            # modify this 10 to every dimension you want.
            relation_1 = relation_1.expand(-1, -1, -1, -1, 10)   # (batch_size, 2, 2, rank/2, 10)
            entity_2 = torch.einsum('ijklh,ikmlh->ijmlh', relation_1, entity_1)   # (batch_size, 2, 1, rank/2, 10)
    
            return entity_2.reshape(entity_2.size(0), -1)  # (batch_size, 10 * rank)    return transformed entity
        
        lhs_e = _matrix_transformation(self.entity(queries[:, 0]), self.rel(queries[:, 1]))
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
"""