import os
import time
import torch
import argparse
import numpy as np
from model import *
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
parser.add_argument('--emb_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--dropout_rate', default=0.7, type=float)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'lastfm':
        n_node = 9998
    elif opt.dataset == 'gowalla':
        n_node = 37722
    else:
        n_node = 309
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=False, n_node=n_node)
    model = trans_to_cuda(NARM(n_node, opt))
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
        best_results['nDCG%d' % K] = [0, 0, 0]
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['nDCG%d' % K] = np.mean(metrics['nDCG%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                torch.save(model, "../narm_lastfm_res5.pkl")
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['nDCG%d' % K]:
                best_results['metric%d' % K][2] = metrics['nDCG%d' % K]
                best_results['epoch%d' % K][2] = epoch
                torch.save(model, "../narm_lastfm_res5.pkl")
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tnDCG%d: %.4f\tEpoch: %d,  %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K,
                   best_results['metric%d' % K][2],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))


if __name__ == '__main__':
    main()
