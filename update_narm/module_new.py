import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


class CODE_AE(nn.Module):

    def __init__(self, args):
        super(CODE_AE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x, codebook1, codebook2, one_hot1, codebook3, codebook4):
        y_w = self.encoder(x)
        # one_hot = torch.cat([y_w, one_hot1], dim=0)
        rec_emb = self.decoder(y_w, codebook1, codebook2, codebook3, codebook4)

        return y_w, rec_emb


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.cluster_num = args.cluster_num
        self.code_book_len = args.code_book_len
        self.l = args.l

        self.theta = nn.Linear(self.hidden_dim, (self.cluster_num * self.code_book_len) // 2, bias=True)
        self.theta_p = nn.Linear((self.cluster_num * self.code_book_len) // 2, self.cluster_num * self.code_book_len,
                                 bias=True)
        # self.l = args.l
        # self.size = self.cluster_num * self.code_book_len + int((self.cluster_num * self.code_book_len) / self.l)
        # self.theta = nn.Linear(self.hidden_dim, self.size // 2, bias=True)
        # self.theta_p = nn.Linear(self.size // 2, self.size,
        #                          bias=True)

    def forward(self, x):
        # x_n = x[:int(x.shape[0]/self.l)]
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
        # self.outx = x
        # shape = x.size()
        # _, ind = x.max(dim=-1)
        # y_hard = torch.zeros_like(x).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # return (y_hard - x).detach() + x

    def _sample_gumbel(self, batch_size, eps=1e-10):
        u = torch.zeros(batch_size, self.code_book_len, self.cluster_num).uniform_().cuda()
        return -1 * torch.log(-1 * torch.log(u + eps) + eps)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.cluster_num = args.cluster_num
        self.code_book_len = args.code_book_len
        self.l = args.l
        # self.size = self.cluster_num * self.code_book_len + int((self.cluster_num * self.code_book_len) / self.l)
        self.A = nn.Parameter(torch.FloatTensor(int((self.code_book_len * self.cluster_num)/self.l), self.hidden_dim),
                              requires_grad=True)
        # self.A = trans_to_cuda(torch.randn(self.code_book_len * self.cluster_num, self.hidden_dim))
        # self.A = nn.Parameter(torch.FloatTensor(self.code_book_len * self.cluster_num, self.hidden_dim),
        #                       requires_grad=True)

    def forward(self, y_w, codebook1, codebook2, codebook3, codebook4):
        y_w = y_w.reshape(-1, self.cluster_num * self.code_book_len)  # B x MK
        codebook = torch.cat([codebook1, self.A], dim=0)
        # codebook = torch.cat([self.A, codebook1[int((self.code_book_len * self.cluster_num)/self.l):]], dim=0)
        # codebook = torch.cat([codebook1, self.A], dim=0)
        # shape_dim1 = codebook2.shape[0]
        # size = shape_dim1 + int((self.code_book_len * self.cluster_num)/self.l)
        # codebook = torch.cat([codebook3, codebook2, codebook1], dim=0)
        # size = self.cluster_num * self.code_book_len - int((self.code_book_len * self.cluster_num)/self.l)
        # codebook = torch.cat([self.A, codebook[:size]], dim=0)
        # codebook = torch.cat([codebook2, codebook1], dim=0)
        # size = self.cluster_num * self.code_book_len - int((self.code_book_len * self.cluster_num) / self.l)
        # codebook = torch.cat([self.A, codebook[:size]], dim=0)
        # codebook = torch.cat([codebook4, codebook3, codebook2, codebook1], dim=0)
        # size = self.cluster_num * self.code_book_len - int((self.code_book_len * self.cluster_num) / self.l)
        # codebook = torch.cat([self.A, codebook[:size]], dim=0)
        E_hat = torch.matmul(y_w, codebook)  # B x MK, MK x h -> B x H
        return E_hat


# class Classifier(nn.Module):
#     def __init__(self, args, glove_embedding):
#         super(Classifier, self).__init__()
#         self.vocab_size, self.emb_size = glove_embedding.size()
#         self.embedding = nn.Embedding(self.vocab_size, self.emb_size).from_pretrained(glove_embedding)
#         self.embedding.weight.requires_grad = False
#
#         self.lstm = nn.LSTM(input_size=self.emb_size, hidden_size=args.hidden_size, batch_first=True)
#         self.fc = nn.Linear(args.hidden_size, 2)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         out, _ = self.lstm(x)  # B H, (c,h)
#         out = self.fc(out[:, -1, :])
#         logits = F.log_softmax(out, dim=1)
#
#         return logits