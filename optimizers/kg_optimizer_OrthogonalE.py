"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn


class KGOptimizer_OrthogonalE(object):
    """Knowledge Graph embedding model optimizer.

    KGOptimizers performs loss computations with negative sampling and gradient descent steps.

    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
        neg_sample_size: An integer for the number of negative samples
        double_neg: A boolean (True to sample both head and tail entities)
    """

    def __init__(
            self, model, regularizer, optim_relation, optim_entity, optim_bh, optim_bt, batch_size, neg_sample_size, double_neg, verbose=True):
        """Inits KGOptimizer."""
        self.model = model
        self.regularizer = regularizer

        self.optim_relation = optim_relation
        self.optim_entity = optim_entity
        self.optim_bh = optim_bh
        self.optim_bt = optim_bt

        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.module.sizes[0]

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optim_relation.param_groups:
            param_group['lr'] *= factor
        for param_group in self.optim_entity.param_groups:
            param_group['lr'] *= factor
        for param_group in self.optim_bh.param_groups:
            param_group['lr'] *= factor
        for param_group in self.optim_bt.param_groups:
            param_group['lr'] *= factor

    def get_neg_samples(self, input_batch):
        """Sample negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        batch_size = input_batch.shape[0]
        negsamples = torch.Tensor(np.random.randint(
            self.n_entities,
            size=batch_size * self.neg_sample_size)
        ).to(input_batch.dtype)
        negative_batch[:, 2] = negsamples
        if self.double_neg:
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=batch_size * self.neg_sample_size)
            ).to(input_batch.dtype)
            negative_batch[:, 0] = negsamples
        return negative_batch

    
    """  
    # This the adversial negative sampling. You can try this to train FB15K-237 if you cannot get the best result.

    def neg_sampling_loss(self, input_batch):   # input_batch: (batch_size, 3)  修改后的版本。
        # 这里有一个问题，negative sample有很大的维数，但是positive 只有一个，mean之后，positive的loss就会被稀释。

        # positive samples
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(positive_score)   # (batch_size, 1)
        positive_score = positive_score.squeeze(dim = 1) # (batch_size)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = negative_score.reshape(-1, self.neg_sample_size)  # (batch_size, neg_sample_size)
        neg_softmax = F.softmax(negative_score, dim = 1) # (batch_size, neg_sample_size)
        negative_score = neg_softmax.detach() * F.logsigmoid(-negative_score)  # (batch_size, neg_sample_size)
        negative_score = negative_score.sum(dim = 1)  # (batch_size)
        # .mean() 不规定维度，就是求所有元素的均值
        #loss = - torch.cat([positive_score, negative_score], dim=0).mean()   # (batch_size * neg_sample_size + batch_size, 1)
        loss = - (positive_score + negative_score) / 2   # (batch_size)
        loss = loss.mean()  # (1)
        return loss, factors      # loss size: 1, factors size: (n_entities, d)
    """ 

    # This is the uniform negative sampling.
    def neg_sampling_loss(self, input_batch):
        
        
        # positive samples
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(positive_score)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        return loss, factors
    
    
    def no_neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss without negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        predictions, factors = self.model(input_batch, eval_mode=True)
        truth = input_batch[:, 2]
        log_prob = F.logsigmoid(-predictions)
        idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
        pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
        log_prob[idx, truth] += pos_scores
        loss = - log_prob.mean()
        loss += self.regularizer.forward(factors)
        return loss, factors
    

    def calculate_loss(self, input_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)
            # loss, factors = self.no_neg_sampling_loss(input_batch)

        # regularization loss
        loss += self.regularizer.forward(factors)
        return loss

    def calculate_valid_loss(self, examples):
        """Compute KG embedding loss over validation examples.

        Args:
            examples: torch.LongTensor of shape (N_valid x 3) with validation triples

        Returns:
            loss: torch.Tensor with loss averaged over all validation examples
        """
        b_begin = 0
        loss = 0.0
        counter = 0
        with torch.no_grad():
            while b_begin < examples.shape[0]:
                input_batch = examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                b_begin += self.batch_size
                loss += self.calculate_loss(input_batch)
                counter += 1
        loss /= counter
        return loss

    def epoch(self, examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            total_loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()

                # gradient step
                l = self.calculate_loss(input_batch)
                self.optim_relation.zero_grad()
                self.optim_entity.zero_grad()
                self.optim_bh.zero_grad()
                self.optim_bt.zero_grad()
                l.backward()
                self.optim_relation.step()
                self.optim_entity.step()
                self.optim_bh.step()
                self.optim_bt.step()
                
                b_begin += self.batch_size
                total_loss += l
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')
        total_loss /= counter
        return total_loss
