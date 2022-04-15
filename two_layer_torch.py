# -*- coding: utf-8 -*-
# @Time : 2022/4/10 16:35
# @Author : hhq
# @File : two_layer_torch.py
import torch
import numpy as np
import cupy as cp
import time
start_time = time.time()
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# 种群规模为m,交叉概率0.8，变异概率0.2
m, cr, mu, sr = 20, 0.8, 0.2, 0.7
#
w1 = torch.randn(m, D_in, H)  # 每个染色体对应的输入和第一层的连接权重
w2 = torch.randn(m, H, D_out)  # 染色体对应的第一层和和第二层的连接权重
b1 = torch.randn(m, N)  # 第一层的阈值
b2 = torch.randn(m, N)  # 第二层阈值
# 暂时忽略偏置
learning_rate = 1e-6
use_gpu = torch.cuda.is_available()
use_gpu = 0
def use_GPU(v):
    if use_gpu:
        v = v.cuda()
    else:
        v = v
    return v
def use_cupy():
    if use_gpu:
        pp = cp
    else:
        pp = np
    return pp
x, y = use_GPU(x), use_GPU(y)
w1, w2, b1, b2 = use_GPU(w1), use_GPU(w2), use_GPU(b1), use_GPU(b2)
pp = use_cupy()  # gpu_np

def train(w11, w12, b11, b12):
    w11, w12, b11, b12 = use_GPU(w11), use_GPU(w12), use_GPU(b11), use_GPU(b12)
    for it in range(500):
        h = x.mm(w11)
        for i in range(H):
            h[:, i] += b11
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w12)
        for j in range(D_out):
            y_pred[:, j] += b12
        # 要将tensor转为单个数字
        loss = (y_pred - y).pow(2).sum().item()
        # print(it, loss)

        grad_y_pred = 2.0 * (y_pred - y)  # 关于目标值的梯度
        grad_w12 = h_relu.t().mm(grad_y_pred)  # 关于第二层权重的梯度
        grad_b12 = grad_y_pred[:, 0]
        grad_h_relu = grad_y_pred.mm(w12.t())  # 关于第一层输出的激活函数的权重
        grad_h = grad_h_relu.clone()  # 关于第一层输出的权重
        grad_b11 = grad_h[:, 0]
        grad_h[h < 0] = 0  # 激活函数relu
        grad_w11 = x.t().mm(grad_h)  # 第一层连接权重的梯度

        w11 -= learning_rate * grad_w11
        w12 -= learning_rate * grad_w12
        b11 -= learning_rate * grad_b11
        b12 -= learning_rate * grad_b12
    return loss


def selection(fits):
    fit_sum = fits.sum()  # 求出总的tensor
    w1_new = torch.randn(m, D_in, H)  # 每个染色体对应的输入和第一层的连接权重
    w2_new = torch.randn(m, H, D_out)  # 染色体对应的第一层和和第二层的连接权重
    b1_new = torch.randn(m, N)  # 第一层的阈值
    b2_new = torch.randn(m, N)   # 第二层阈值
    # 计算每个个体保留概率
    pi = torch.zeros(m)  # 初始化
    for i in range(m):
        pi[i] = fits[i] / fit_sum
        if pi[i] < pp.random.rand():  # 较优个体保存
            w1_new[i] = w1[i]
            w2_new[i] = w2[i]
            b1_new[i] = b1[i]
            b2_new[i] = b2[i]
    return w1_new, w2_new, b1_new, b2_new  # 新的种群


def cross(w1, w2, b1, b2):  # w1_new, w2_new, b1_new, b2_new均为种群
    for i in range(0, m, 2):  # 判断两染色体是否交叉
        if cr < pp.random.rand():
            # 交叉长度
            cr_l1 = pp.random.randint(1, len(w1[i]) - 1)
            crp1 = w1[i][:, :cr_l1]
            w1[i][:, :cr_l1] = w1[i + 1][:, :cr_l1]
            w1[i + 1][:, :cr_l1] = crp1

            cr_l2 = pp.random.randint(1, len(w2[i]) - 1)
            crp2 = w2[i][:, :cr_l2]
            w2[i][:, :cr_l2] = w2[i + 1][:, :cr_l2]
            w2[i + 1][:, :cr_l2] = crp2

            cr_l3 = pp.random.randint(1, len(b1[i]) - 1)
            crp3 = b1[i][:cr_l3]
            b1[i][:cr_l3] = b1[i + 1][:cr_l3]
            b1[i + 1][:cr_l3] = crp3

            cr_l4 = pp.random.randint(1, len(b2[i]) - 1)
            crp4 = b2[i][:cr_l4]
            b2[i][:cr_l4] = b2[i + 1][:cr_l4]
            b2[i + 1][:cr_l4] = crp4

    return w1, w2, b1, b2


def mutation(w1, w2, b1, b2):
    for i in range(m):  # 个体是否变异
        if pp.random.rand() > mu:
            # 变异点
            l1 = pp.random.randint(0, len(w1[i]) - 1)
            w1[i][l1] = torch.randn(H)

            l2 = pp.random.randint(0, len(w2[i]) - 1)
            w2[i][l2] = torch.randn(D_out)

            l3 = pp.random.randint(0, len(b1[i]) - 1)
            b1[i][l3] = torch.randn(1)

            l4 = pp.random.randint(0, len(b2[i]) - 1)
            b2[i][l4] = torch.randn(1)

    return w1, w2, b1, b2


W1_best, w2_best, b1_best, b2_best = w1, w2, b1, b2  # 初始化
iters = 2
for it in range(iters):  # 开始迭代
    fits = torch.zeros(m)  # 适应度初始化
    for i in range(m):
        fits[i] = train(W1_best[i], w2_best[i], b1_best[i], b2_best[i])
    if it == iters - 1:
        break
    else:
        fits = use_GPU(fits)
        w1_new, w2_new, b1_new, b2_new = selection(fits)

        ww1, ww2, bb1, bb2 = cross(w1_new, w2_new, b1_new, b2_new)
        W1_best, w2_best, b1_best, b2_best = mutation(ww1, ww2, bb1, bb2)
end_time = time.time()
time_spend = end_time - start_time
use = '不使用'
if use_gpu:
    fits = fits.cpu()
    use = '使用'
print(use + 'GPU花费的时间为：' + str(time_spend) + 's')
fits = fits.numpy()
index_minimum = np.where(fits == np.min(fits))
print(np.min(fits))
print(index_minimum)