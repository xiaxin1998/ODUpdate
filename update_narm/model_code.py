import datetime
import numpy as np
from numba import jit
import heapq
from module import *



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



import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARMRec(nn.Module):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self, n_node, config):
        super(NARMRec, self).__init__()

        # load parameters info
        self.n_items = n_node

        self.hidden_dim = config.hidden_dim
        self.cluster_num = config.cluster_num
        self.code_book_len = config.code_book_len
        self.batch_size = config.batch_size

        self.theta = nn.Linear(self.hidden_dim, (self.cluster_num * self.code_book_len) // 2, bias=True)
        self.theta_p = nn.Linear((self.cluster_num * self.code_book_len) // 2, self.cluster_num * self.code_book_len,
                                 bias=True)
        self.mlp = nn.Linear((self.cluster_num * self.code_book_len), (self.cluster_num * self.code_book_len), bias=True)
        self.A = nn.Parameter(torch.FloatTensor(self.code_book_len * self.cluster_num, self.hidden_dim))

        self.loss_fct = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def _sample_gumbel(self, batch_size, eps=1e-10):
        u = torch.zeros(batch_size, self.code_book_len, self.cluster_num).uniform_().cuda()
        return (-1 * torch.log(-1 * torch.log(u + eps) + eps)) / 10.0

    def _gumbel_softmax(self, x, eps=1e-10, train=True):  # B x m x k -> B x m x k (one-hot), tau=1
        x = torch.log(x + eps)
        if train:
            g = self._sample_gumbel(x.size(0))
            x = x + Variable(g)
        x = F.softmax(x / 0.3, dim=2)
        return x

    def forward(self, teacher):
        teacher.eval()
        h_w = torch.tanh(self.theta(teacher.item_embedding.weight))  # B x mk/2
        a_w = F.softplus(self.theta_p(h_w))  # B x m*k
        a_w = a_w.reshape(-1, self.code_book_len, self.cluster_num)  # B x m x k
        self.a_w = F.softmax(a_w, dim=2)
        y_w = self._gumbel_softmax(self.a_w, train=True)
        self.embedding = torch.matmul(y_w.reshape(-1, self.code_book_len * self.cluster_num), self.A)
        mse_loss = self.loss_fct(self.embedding, teacher.item_embedding.weight).div(len(teacher.item_embedding.weight))
        return mse_loss

    def forward_test(self, item_seq, item_seq_len, teacher):
        teacher.eval()
        y_w = self._gumbel_softmax(self.a_w, train=False)
        self.embedding = torch.matmul(y_w.reshape(-1, self.code_book_len * self.cluster_num), self.A)

        item_seq_emb = self.embedding[item_seq]
        item_seq_emb_dropout = teacher.emb_dropout(item_seq_emb)
        gru_out, _ = teacher.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = teacher.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = teacher.a_1(gru_out)
        q2 = teacher.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = teacher.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = teacher.ct_dropout(c_t)
        seq_output = teacher.b(c_t)

        logits = torch.matmul(seq_output, self.embedding.transpose(0, 1))
        return logits



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


def train_test(model, train_data, test_data, epoch, opt, teacher):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    cnt = 0
    slices = train_data.generate_batch(opt.batch_size)
    for i in slices:
        mse = model(teacher)
        loss = mse
        model.zero_grad()
        loss.backward()
        if cnt % 100 == 0:
            print(loss)
        cnt += 1
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
        items, tar, session_len, mask, last = test_data.get_slice(i)
        session_item = trans_to_cuda(torch.Tensor(items).long())
        seq_len = trans_to_cuda(torch.Tensor(session_len).long())

        score = model.forward_test(session_item, seq_len, teacher)
        # score = teacher(session_item, seq_len)
        scores = trans_to_cpu(score).detach().numpy()
        index = []
        for idd in range(100):
            index.append(find_k_largest(20, scores[idd]))
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
        # print(metrics)
    return metrics, total_loss






