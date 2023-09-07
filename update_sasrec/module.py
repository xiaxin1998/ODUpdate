import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable


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

# class CODE_AE(nn.Module):
#
#     def __init__(self, args):
#         super(CODE_AE, self).__init__()
#         self.encoder = Encoder(args)
#         self.decoder = Decoder(args)
#
#     def forward(self, x):
#         y_w = self.encoder(x)
#         rec_emb = self.decoder(y_w)
#
#         return y_w, rec_emb


class Encoder(nn.Module):
    def __init__(self, args, n_node):
        super(Encoder, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.cluster_num = args.cluster_num
        self.code_book_len = args.code_book_len

        self.theta = nn.Linear(self.hidden_dim, (self.cluster_num * self.code_book_len) // 2, bias=True)
        self.theta_p = nn.Linear((self.cluster_num * self.code_book_len) // 2, self.cluster_num * self.code_book_len,
                                 bias=True)

    def forward(self, x):
        h_w = torch.tanh(self.theta(x))  # B x mk/2
        a_w = F.softplus(self.theta_p(h_w))  # B x m*k
        a_w = a_w.reshape(-1, self.code_book_len, self.cluster_num)  # B x m x k
        a_w = F.softmax(a_w, dim=2)
        self.a_w = a_w
        y_w = self._gumbel_softmax(a_w)  # B x m x k

        return y_w

    def _gumbel_softmax(self, x, eps=1e-10):  # B x m x k -> B x m x k (one-hot), tau=1
        x = torch.log(x + eps)
        g = self._sample_gumbel(x.size(0))
        x = x + Variable(g)
        x = F.softmax(x / 0.1, dim=2)
        return x


    def _sample_gumbel(self, batch_size, eps=1e-10):
        u = torch.zeros(batch_size, self.code_book_len, self.cluster_num).uniform_().cuda()
        return -1 * torch.log(-1 * torch.log(u + eps) + eps)


class Decoder(nn.Module):
    def __init__(self, args, n_node):
        super(Decoder, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.cluster_num = args.cluster_num
        self.code_book_len = args.code_book_len
        self.A = trans_to_cuda(torch.randn(self.code_book_len * self.cluster_num, self.hidden_dim, requires_grad=True))
        # self.A = nn.Parameter(torch.FloatTensor(self.code_book_len * self.cluster_num, self.hidden_dim))

    def forward(self, y_w):
        y_w = y_w.reshape(-1, self.code_book_len * self.cluster_num)  # B x MK
        E_hat = torch.matmul(y_w, self.A)  # B x MK, MK x h -> B x H
        return E_hat

