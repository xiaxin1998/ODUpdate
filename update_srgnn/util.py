import sys
import copy
import random
import numpy as np
from collections import defaultdict
from operator import itemgetter


def count_hot(sessions, target):
    cnt_hot = {}
    for sess in sessions:
        for i in sess:
            if i not in cnt_hot:
                cnt_hot[i] = 1
            else:
                cnt_hot[i] += 1
    cnt_hot = sorted(cnt_hot.items(), key=lambda kv: kv[1], reverse=True)
    # hot_item = list(cnt_hot.keys())
    length = len(cnt_hot)
    hot = [cnt_hot[i][0] for i in range(int(length*0.2))]
    most_pop = [cnt_hot[i][0] for i in range(int(length*0.05))]
    return cnt_hot, hot, most_pop

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Data():
    def __init__(self, data, train_data, shuffle=False, n_node=None, train=None):
        # self.lastfm = np.asarray(data[0])
        self.n_node = n_node
        # self.targets = np.asarray(data[1])
        # self.length = len(self.lastfm)
        # self.shuffle = shuffle
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        # self.graph = graph
        # if train == 1:
        #     self.item_dict, self.hot_item, self.most_pop = count_hot(data[0], data[1])
        # else:
        #     self.item_dict, self.hot_item, self.most_pop = train_data.item_dict, train_data.hot_item, train_data.most_pop

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets