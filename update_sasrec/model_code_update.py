import copy
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
from numba import jit
import heapq
from module_new import *
import random


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    def __init__(self):
        # self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError



    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)


class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """
    # type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, n_node, config):
        super(SASRec, self).__init__(n_node, config)

        # load parameters info
        self.n_items = n_node
        self.hidden_size = config.hidden_units  # same as embedding_size
        self.emb_size = config.hidden_units
        self.inner_size = config.inner_units # the dimensionality in feed-forward layer
        self.hidden_dim = config.hidden_dim
        self.cluster_num = config.cluster_num
        self.code_book_len = config.code_book_len
        self.l = config.l

        self.theta = nn.Linear(self.hidden_dim, (self.cluster_num * self.code_book_len) // 2, bias=True)
        self.theta_p = nn.Linear((self.cluster_num * self.code_book_len) // 2, self.cluster_num * self.code_book_len,
                                 bias=True)
        # self.mlp = nn.Linear((self.cluster_num * self.code_book_len), (self.cluster_num * self.code_book_len), bias=True)
        self.A = nn.Parameter(torch.FloatTensor(int((self.code_book_len * self.cluster_num)/self.l), self.hidden_dim))
        # self.A = nn.Linear(self.code_book_len * self.cluster_num, self.hidden_dim, bias=False)
        self.batch_size = config.batch_size
        self.initializer_range = 0.01
        self.loss_fct = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(0.1)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)
            # weight.data.normal_(0, 0.1)

    def _sample_gumbel(self, batch_size, eps=1e-10):
        u = torch.zeros(batch_size, self.code_book_len, self.cluster_num).uniform_().cuda()
        return (-1 * torch.log(-1 * torch.log(u + eps) + eps)) / 10.0

    def _gumbel_softmax(self, x, eps=1e-10, train=True):  # B x m x k -> B x m x k (one-hot), tau=1
        x = torch.log(x + eps)
        if train:
            g = self._sample_gumbel(x.size(0))
            x = x + Variable(g)
        x = F.softmax(x / 0.5, dim=2)
        return x

    def forward(self, teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4):
        teacher1.eval()
        teacher2.eval()
        teacher3.eval()
        teacher4.eval()
        teacher5.eval()
        code_model1.eval()
        code_model2.eval()
        code_model3.eval()
        code_model4.eval()
        # codebook = code_model3.codebook
        codebook = code_model1.A.weight.data.reshape(-1, self.hidden_size)
        size = self.code_book_len * self.cluster_num - int((self.code_book_len * self.cluster_num)/self.l)
        self.codebook = torch.cat([codebook[-size:], self.A], dim=0)   # queue-based method
        #self.codebook = torch.cat([codebook[:size], self.A], dim=0)     # stack-based method
        h_w = torch.tanh(self.theta(teacher2.embedding.weight))  # B x mk/2
        a_w = F.softplus(self.theta_p(h_w))  # B x m*k
        a_w = a_w.reshape(-1, self.code_book_len, self.cluster_num)  # B x m x k
        self.a_w = F.softmax(a_w, dim=2)
        y_w = self._gumbel_softmax(self.a_w, train=True)

        self.embedding = torch.matmul(y_w.reshape(-1, self.code_book_len * self.cluster_num), self.codebook)
        mse_loss = self.loss_fct(self.embedding, teacher2.embedding.weight).div(len(teacher2.embedding.weight))
        return mse_loss

    def forward_test(self, item_seq, item_seq_len, mask, teacher):
        teacher.eval()

        y_w = self._gumbel_softmax(self.a_w, train=False)
        self.embedding = torch.matmul(y_w.reshape(-1, self.code_book_len * self.cluster_num), self.codebook)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = teacher.position_embedding(position_ids)

        item_emb = self.embedding[item_seq]
        item_emb = item_emb.reshape(self.batch_size, -1, self.emb_size)
        input_emb = item_emb + position_embedding
        input_emb = teacher.LayerNorm(input_emb)
        input_emb = teacher.dropout(input_emb)
        extended_attention_mask = teacher.get_attention_mask(item_seq)

        trm_output = teacher.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = teacher.gather_indexes(output, item_seq_len - 1)
        return output, self.embedding

    def full_sort_predict(self, item_seq, seq_len, mask, teacher):
        seq_output, item_emb = self.forward_test(item_seq, seq_len, mask, teacher)
        scores = torch.matmul(seq_output, item_emb.transpose(0, 1))
        return scores


def forward(model, i, data, teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4):
    mse_loss = model(teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4)
    return mse_loss


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    # k_largest_scores = [item[0] for item in n_candidates]
    return ids#, k_largest_scores

#
# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def train_test(model, train_data, test_data, epoch, opt, teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4):

    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    # set_seed(opt)
    slices = train_data.generate_batch(opt.batch_size)

    cnt = 0
    for i in slices:
        cnt += 1
        # print(model.emb_model.decoder.A)
        mse_loss = forward(model, i, train_data, teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4)
        loss = mse_loss
        model.zero_grad()
        loss.backward()
        if cnt % 100 == 0:
            print(loss)
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['nDCG%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(opt.batch_size)
    for i in slices:
        session_item, tar, seq_len, mask = test_data.get_slice(i)
        session_item = trans_to_cuda(torch.Tensor(session_item).long())
        seq_len = trans_to_cuda(torch.Tensor(seq_len).long())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        score = model.full_sort_predict(session_item, seq_len, mask, teacher2)
        scores = trans_to_cpu(score).detach().numpy()
        index = []
        for idd in range(100):
            index.append(find_k_largest(20, scores[idd]))
        del score, scores
        index = np.array(index)
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['nDCG%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['nDCG%d' % K].append(np.log(2) * 1 / (np.log(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss
