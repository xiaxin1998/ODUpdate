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
parser.add_argument('--act', default='gelu')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--inner_units', default=128, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--seed', type=int, default=55)
opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train5_20.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'gowalla':
        n_node = 37723
    elif opt.dataset == 'lastfm':
        n_node = 9997
    else:
        n_node = 309 + 1
    train_data = Data(train_data, train_data, shuffle=True, n_node=n_node, train=1)
    test_data = Data(test_data, train_data, shuffle=False, n_node=n_node, train=0)
    model = SAS(n_node, opt)
    model = trans_to_cuda(model)
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
        best_results['nDCG%d' % K] = [0, 0, 0]

    for epoch in range(200):
        print('------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['nDCG%d' % K] = np.mean(metrics['nDCG%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                torch.save(model, '../sas_lastfm_res5.pkl')
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['nDCG%d' % K]:
                best_results['metric%d' % K][2] = metrics['nDCG%d' % K]
                best_results['epoch%d' % K][2] = epoch
                torch.save(model, '../sas_lastfm_res5.pkl')
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tnDCG%d: %.4f\tEpoch: %d,  %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K,
                   best_results['metric%d' % K][2],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))



if __name__ == '__main__':
    main()
