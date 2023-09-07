import argparse
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

#digi: 1 2 1 0.5
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--inner_units', default=128, type=int)
parser.add_argument('--hidden_dim',      type=int,   default=128,      help='number of dimensions of input size')
parser.add_argument('--code_book_len',   type=int,   default=40,       help='number of codebooks')
parser.add_argument('--cluster_num',     type=int,   default=32,       help='length of a codebook')
parser.add_argument('--l', type=int, default=10, help='update compression ratio')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train2_12345.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'gowalla':
        n_node = 37723
    elif opt.dataset == 'lastfm':
        n_node = 9997
    else:
        n_node = 309 + 1
    print('Compression Ratio:', (n_node*opt.hidden_dim)/(n_node*opt.code_book_len + opt.code_book_len*opt.cluster_num*opt.hidden_dim))
    train_data = Data(train_data, train_data, shuffle=True, n_node=n_node, train=1)
    test_data = Data(test_data, train_data, shuffle=False, n_node=n_node, train=0)
    model = SASRec(n_node, opt)
    model = trans_to_cuda(model)
    # the following models should be pretrained and stored
    teacher1 = torch.load('../sas_gowalla_res1.pkl')
    teacher2 = torch.load('../sas_gowalla_res2.pkl')
    teacher3 = torch.load('../sas_gowalla_res3.pkl')
    teacher4 = torch.load('../sas_gowalla_res4.pkl')
    teacher5 = torch.load('../sas_gowalla_res5.pkl')
    # the following models below should be pretrained and stored at each time slice sequentially
    # sas_gowalla_res1_40m_32k means the SASRec model is trained on gowalla in T1 with m=40 and k=32
    code_model1 = torch.load("../sas_gowalla_res1_40m_32k.pkl")
    code_model2 = torch.load("../sas_gowalla_res2_40m_32k.pkl")
    code_model3 = torch.load("../sas_gowalla_res3_40m_32k.pkl")
    code_model4 = torch.load("../sas_gowalla_res4_40m_32k.pkl")

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
        best_results['nDCG%d' % K] = [0, 0, 0]
    for epoch in range(200):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        print(opt.l)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt, teacher1, teacher2, teacher3, teacher4, teacher5, code_model1, code_model2, code_model3, code_model4)

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['nDCG%d' % K] = np.mean(metrics['nDCG%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                torch.save(model, '../sas_gowalla_res2_40m_32k_l10_queue.pkl')
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['nDCG%d' % K]:
                best_results['metric%d' % K][2] = metrics['nDCG%d' % K]
                best_results['epoch%d' % K][2] = epoch
                torch.save(model, '../sas_gowalla_res2_40m_32k_l10_queue.pkl')
        print(metrics)
        print(opt.l)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tnDCG%d: %.4f\tEpoch: %d,  %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K,
                   best_results['metric%d' % K][2],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))



if __name__ == '__main__':
    main()
