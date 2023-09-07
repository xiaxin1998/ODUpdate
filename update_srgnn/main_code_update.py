import os
import time
import torch
import argparse
import numpy as np
from model_code_update import *
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
parser.add_argument('--dataset', default='gowalla')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--hidden_dim',      type=int,   default=128,      help='number of dimensions of input size')
parser.add_argument('--code_book_len',   type=int,   default=20,       help='number of codebooks')
parser.add_argument('--cluster_num',     type=int,   default=32,       help='length of a codebook')
parser.add_argument('--l', type=int, default=13, help='update compression ratio')
opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train3_12345.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'gowalla':
        n_node = 37722
    elif opt.dataset == 'lastfm':
        n_node = 9998
    else:
        n_node = 309 + 1
    print('Compression Ratio:', (n_node * opt.hidden_dim) / (n_node * opt.code_book_len + opt.code_book_len * opt.cluster_num * opt.hidden_dim))
    train_data = Data(train_data, train_data, shuffle=True, n_node=n_node, train=1)
    test_data = Data(test_data, train_data, shuffle=False, n_node=n_node, train=0)
    model = trans_to_cuda(SessionGraph(opt, n_node))
    teacher1 = torch.load('../srgnn_gowalla_res1.pkl')
    teacher2 = torch.load('../srgnn_gowalla_res2.pkl')
    teacher3 = torch.load('../srgnn_gowalla_res3.pkl')
    teacher4 = torch.load('../srgnn_gowalla_res4.pkl')
    teacher5 = torch.load('../srgnn_gowalla_res5.pkl')
    code_model1 = torch.load("../srgnn_gowalla_res1_20m_32k.pkl")
    code_model2 = torch.load("../srgnn_gowalla_res2_20m_32k_l11_queue.pkl")
    code_model3 = torch.load("../srgnn_lastfm_res3_20m_32k_l10_queue.pkl")
    code_model4 = torch.load("../srgnn_lastfm_res4_20m_32k_l10_queue.pkl")
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
        best_results['nDCG%d' % K] = [0, 0, 0]

    for epoch in range(200):
        print('------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt, teacher1, teacher2, teacher3,
                                         teacher4, teacher5, code_model1, code_model2, code_model3, code_model4)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['nDCG%d' % K] = np.mean(metrics['nDCG%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                torch.save(model, '../srgnn_gowalla_res3_20m_32k_l13_queue.pkl')
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['nDCG%d' % K]:
                best_results['metric%d' % K][2] = metrics['nDCG%d' % K]
                best_results['epoch%d' % K][2] = epoch
                torch.save(model, '../srgnn_gowalla_res3_20m_32k_l13_queue.pkl')
        print(metrics)
        print(opt.l)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tnDCG%d: %.4f\tEpoch: %d,  %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K,
                   best_results['metric%d' % K][2],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))



if __name__ == '__main__':
    main()
