import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from numba import jit
from module_new import *
import heapq
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
#
class GNN(Module):
    def __init__(self, hidden_size):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(1):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.hidden_dim = opt.hidden_dim
        self.cluster_num = opt.cluster_num
        self.code_book_len = opt.code_book_len
        self.l = opt.l
        self.theta = nn.Linear(self.hidden_dim, (self.cluster_num * self.code_book_len) // 2, bias=True)
        self.theta_p = nn.Linear((self.cluster_num * self.code_book_len) // 2, self.cluster_num * self.code_book_len,
                                 bias=True)
        self.mlp = nn.Linear((self.cluster_num * self.code_book_len), (self.cluster_num * self.code_book_len),
                             bias=True)
        self.A = nn.Parameter(torch.FloatTensor(int((self.code_book_len * self.cluster_num)/self.l), self.hidden_dim))
        self.mse = nn.MSELoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.normal_(0, 0.1)

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
        # self.codebook1 = code_model1.A.data.reshape(-1, self.hidden_dim)
        self.codebook1 = code_model2.codebook
        size = self.code_book_len * self.cluster_num - int((self.code_book_len * self.cluster_num) / self.l)
        self.codebook = torch.cat([self.codebook1[-size:], self.A], dim=0)
        h_w = torch.tanh(self.theta(teacher3.embedding.weight))  # B x mk/2
        a_w = F.softplus(self.theta_p(h_w))  # B x m*k
        a_w = a_w.reshape(-1, self.code_book_len, self.cluster_num)  # B x m x k
        self.a_w = F.softmax(a_w, dim=2)
        y_w = self._gumbel_softmax(self.a_w, train=True)

        self.embedding = torch.matmul(y_w.reshape(-1, self.code_book_len * self.cluster_num), self.codebook)
        mse_loss = self.mse(self.embedding, teacher3.embedding.weight).div(len(teacher3.embedding.weight))
        return mse_loss

    def forward_test(self, alias_inputs, A, items, mask, teacher):
        teacher.eval()
        y_w = self._gumbel_softmax(self.a_w, train=False)
        self.embedding = torch.matmul(y_w.reshape(-1, self.code_book_len * self.cluster_num), self.codebook)
        hidden = self.embedding[items]
        hidden = teacher.gnn(A, hidden)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        logits = teacher.compute_scores(seq_hidden, mask)
        return logits

#
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def train_test(model, train_data, test_data, epoch, opt, teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batchSize)
    cnt = 0
    model.train()
    for i in slices:
        cnt += 1
        mse = model(teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4)
        loss = mse
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
    slices = test_data.generate_batch(opt.batchSize)
    for i in slices:
        alias_inputs, A, items, mask, targets = test_data.get_slice(i)
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(A).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        scores = model.forward_test(alias_inputs, A, items, mask, teacher3)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(100):
            index.append(find_k_largest(20, scores[idd]))
        del scores
        index = np.array(index)
        for K in top_K:
            for prediction, target in zip(index[:, :K], targets-1):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['nDCG%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['nDCG%d' % K].append(np.log(2) * 1 / (np.log(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss