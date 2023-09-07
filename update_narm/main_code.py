import os
import time
import torch
import argparse
import numpy as np
from model_code import *
from util import *
import pickle


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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lastfm')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--hidden_dim',      type=int,   default=128,      help='number of dimensions of input size')
parser.add_argument('--code_book_len',   type=int,   default=40,       help='number of codebooks')
parser.add_argument('--cluster_num',     type=int,   default=32,       help='length of a codebook')
opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train1_20.txt', 'rb'))
    # print(train_data)
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'lastfm':
        n_node = 9998
    elif opt.dataset == 'gowalla':
        n_node = 37722
    else:
        n_node = 309
    print('Compression Ratio:', (n_node * opt.hidden_dim) / (n_node * opt.code_book_len + opt.code_book_len * opt.cluster_num * opt.hidden_dim))
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=False, n_node=n_node)
    model = trans_to_cuda(NARMRec(n_node, opt))
    teacher = torch.load('../narm_lastfm_res1.pkl')

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
        best_results['nDCG%d' % K] = [0, 0, 0]
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt, teacher)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['nDCG%d' % K] = np.mean(metrics['nDCG%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                torch.save(model, "../narm_lastfm_res1_40m_32k.pkl")
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['nDCG%d' % K]:
                best_results['metric%d' % K][2] = metrics['nDCG%d' % K]
                best_results['epoch%d' % K][2] = epoch
                torch.save(model, "../narm_lastfm_res1_40m_32k.pkl")
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tnDCG%d: %.4f\tEpoch: %d,  %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K,
                   best_results['metric%d' % K][2],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))


if __name__ == '__main__':
    main()
