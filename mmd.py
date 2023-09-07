#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.autograd import Variable


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


model1 = torch.load('../srgnn_gowalla_res1_12345.pkl')
model2 = torch.load('../srgnn_gowalla_res2_12345.pkl')
model3 = torch.load('../srgnn_gowalla_res3_12345.pkl')
model4 = torch.load('../srgnn_gowalla_res4_12345.pkl')
model5 = torch.load('../srgnn_gowalla_res5_12345.pkl')
rand1 = torch.randint(high=37721, low=0, size=(1000,))
m1 = mmd(model1.embedding.weight[rand1], model2.embedding.weight[rand1])
print(m1)
m2 = mmd(model2.embedding.weight[rand1], model3.embedding.weight[rand1])
print(m2)
m3 = mmd(model3.embedding.weight[rand1], model4.embedding.weight[rand1])
print(m3)
m4 = mmd(model4.embedding.weight[rand1], model5.embedding.weight[rand1])
print(m4)
# print(mmd(model1.item_embedding.weight[rand1], model5.item_embedding.weight[rand1]))
# print("c=1:")
# print(1 / (2 * torch.sigmoid(torch.sqrt(m1)) - 1))
# print(1 / (2 * torch.sigmoid(torch.sqrt(m2)) - 1))
# print(1 / (2 * torch.sigmoid(torch.sqrt(m3)) - 1))
# print(1 / (2 * torch.sigmoid(torch.sqrt(m4)) - 1))
# print("c=0.1:")
# print(1 / (0.1 *(2 * torch.sigmoid(torch.sqrt(m1)) - 1)))
# print(1 / (0.1 *(2 * torch.sigmoid(torch.sqrt(m2)) - 1)))
# print(1 / (0.1 *(2 * torch.sigmoid(torch.sqrt(m3)) - 1)))
# print(1 / (0.1 *(2 * torch.sigmoid(torch.sqrt(m4)) - 1)))
print("c=0.2:")
print(1 / (0.2 *(2 * torch.sigmoid(torch.sqrt(m1)) - 1)))
print(1 / (0.2 *(2 * torch.sigmoid(torch.sqrt(m2)) - 1)))
print(1 / (0.2 *(2 * torch.sigmoid(torch.sqrt(m3)) - 1)))
print(1 / (0.2 *(2 * torch.sigmoid(torch.sqrt(m4)) - 1)))
# print("c=0.5:")
# print(1 / (0.5 *(2 * torch.sigmoid(torch.sqrt(m1)) - 1)))
# print(1 / (0.5 *(2 * torch.sigmoid(torch.sqrt(m2)) - 1)))
# print(1 / (0.5 *(2 * torch.sigmoid(torch.sqrt(m3)) - 1)))
# print(1 / (0.5 *(2 * torch.sigmoid(torch.sqrt(m4)) - 1)))
# print("c=0.8:")
# print(1 / (0.8 *(2 * torch.sigmoid(torch.sqrt(m1)) - 1)))
# print(1 / (0.8 *(2 * torch.sigmoid(torch.sqrt(m2)) - 1)))
# print(1 / (0.8 *(2 * torch.sigmoid(torch.sqrt(m3)) - 1)))
# print(1 / (0.8 *(2 * torch.sigmoid(torch.sqrt(m4)) - 1)))
