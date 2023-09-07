import sys
import copy
import random
import numpy as np
from collections import defaultdict
import torch

class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        # print(max_n_node)
        session_len = []
        mask = []
        tar = self.targets[index]
        last = []

        for t, session in enumerate(inp):
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            length = len(nonzero_elems)
            sess_items = session + (max_n_node - length) * [0]
            session_len.append([length])
            items.append(sess_items)
            mask.append([1] * length + (max_n_node - length) * [0])
            last.append(session[-1])
        return items, tar, session_len, mask, last

    def collate_fn(data):
        """This function will be used to pad the sessions to max length
           in the batch and transpose the batch from
           batch_size x max_seq_len to max_seq_len x batch_size.
           It will return padded vectors, labels and lengths of each session (before padding)
           It will be used in the Dataloader
        """
        data.sort(key=lambda x: len(x[0]), reverse=True)
        lens = [len(sess) for sess, label in data]
        labels = []
        padded_sesss = torch.zeros(len(data), max(lens)).long()
        for i, (sess, label) in enumerate(data):
            padded_sesss[i, :lens[i]] = torch.LongTensor(sess)
            labels.append(label)

        padded_sesss = padded_sesss.transpose(0, 1)
        return padded_sesss, torch.tensor(labels).long(), lens

