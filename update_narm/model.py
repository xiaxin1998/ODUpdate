import torch
from torch.nn.init import xavier_normal_, constant_
from numba import jit
import datetime
import numpy as np
import heapq


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


class NARM(nn.Module):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self, n_node, config):
        super(NARM, self).__init__()

        # load parameters info
        self.embedding_size = config.emb_size
        self.hidden_size = config.emb_size
        self.n_layers = config.n_layers
        self.dropout_probs = config.dropout_rate

        # define layers and loss
        self.item_embedding = nn.Embedding(n_node, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_probs)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.n_layers, bias=False, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs)
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def _init_weights(self, module):
        for weight in self.parameters():
            # weight.data.normal_(0, 0.1)
            weight.data.uniform_(-0.1, 0.1)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return logits


def forward(model, i, data):
    items, tar, session_len, mask, last = data.get_slice(i)
    items = trans_to_cuda(torch.Tensor(items).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    logits = model(items, session_len)
    loss = model.loss_fct(logits, tar)
    return tar, loss



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


def train_test(model, train_data, test_data, epoch, opt):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batch_size)
    for i in slices:
        tar, loss = forward(model, i, train_data)
        # print(con_loss)
        model.zero_grad()
        loss.backward()
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

        score = model(session_item, seq_len)
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






