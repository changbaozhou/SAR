import argparse
import torch

from utils.smooth_cross_entropy import smooth_crossentropy,trades_loss
from utils.cifar import Cifar,Cifar100
from utils.log import Log
from utils.initialize import initialize
from utils.step_lr import StepLR
from utils.cos_warmup_lr import CosineAnnealingLRWarmup
from utils.Esam import ESAM
from utils.sam import SAM
from torch.utils.tensorboard import SummaryWriter
import os 
# from utils.mail import send_email

from utils.options import args,setup_model
from utils.MiscTools import count_parameters
from utils.dist_util import get_world_size

import torch.nn.functional as F
import logging
from datetime import timedelta
import datetime
# from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from models.CifarResNet import vanilla_resnet20



def load_model(checkpoint):
    model = vanilla_resnet20(10)
    checkpoint = torch.load(checkpoint)
    new_checkpoint = {} ## 新建一个字典来访问模型的权值
    for k,value in checkpoint.items():
        key = k.split('module.')[-1]
        new_checkpoint[key] = value
    checkpoint = new_checkpoint
    model.load_state_dict(checkpoint)
    return model


def main(args):

    checkpoint1 = '/home/bobzhou/SAR/output/resnet20_cifar10_SGD.pth'
    checkpoint2 = '/home/bobzhou/SAR/output/resnet20_cifar10_SAM_rho0.05.pth'



    # Model & Tokenizer Setup
    model1 = load_model(checkpoint1)
    # 获取层的权重
    layer_weights1 = model1.conv_1_3x3.weight.data.cpu()

    weights_mean1 = torch.mean(layer_weights1)
    weights_std1 = torch.std(layer_weights1)

    # 将权重展平为一维数组
    weights_flattened1 = layer_weights1.numpy().flatten()

    print(f'weights_mean1:{weights_mean1}')
    print(f'weights_std1:{weights_std1}')

    model2 = load_model(checkpoint2)

    layer_weights2 = model2.conv_1_3x3.weight.data.cpu()

    weights_mean2 = torch.mean(layer_weights2)
    weights_std2 = torch.std(layer_weights2)

    # 将权重展平为一维数组
    weights_flattened2 = layer_weights2.numpy().flatten()

    print(f'weights_mean2:{weights_mean2}')
    print(f'weights_std2:{weights_std2}')

    # 绘制权重分布和拟合的高斯分布曲线
    plt.figure(figsize=(10, 6))

    # 绘制权重分布直方图（模型1）
    plt.hist(weights_flattened1, bins=50, density=True, alpha=0.5, color='b', label='model1')

    # 计算拟合高斯分布的参数（模型1）
    mu1, std1 = norm.fit(weights_flattened1)
    x_range1 = np.linspace(min(weights_flattened1), max(weights_flattened1), 100)
    pdf1 = norm.pdf(x_range1, mu1, std1)
    plt.plot(x_range1, pdf1, 'r')

    # 绘制权重分布直方图（模型2）
    plt.hist(weights_flattened2, bins=50, density=True, alpha=0.5, color='g', label='model2')

    # 计算拟合高斯分布的参数（模型2）
    mu2, std2 = norm.fit(weights_flattened2)
    x_range2 = np.linspace(min(weights_flattened2), max(weights_flattened2), 100)
    pdf2 = norm.pdf(x_range2, mu2, std2)
    plt.plot(x_range2, pdf2, 'm')

    plt.title('Distribution of First Convolutional Layer Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("weight distribution")



    # # 绘制权重分布的直方图
    # plt.figure(figsize=(10, 6))
    # plt.hist(weights_flattened, bins=50, color='blue', alpha=0.7)
    # # 计算拟合高斯分布的参数
    # mu, std = norm.fit(weights_flattened)
    # # 生成一系列x值来绘制拟合的高斯分布曲线
    # x_range = np.linspace(min(weights_flattened), max(weights_flattened), 100)
    # pdf = norm.pdf(x_range, mu, std)
    # plt.plot(x_range, pdf, 'r', label='Fitted Gaussian')
    # plt.title('Distribution of First Convolutional Layer Weights')
    # plt.xlabel('Weight Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig("weight distribution")
    # plt.show()

    # Training
    # train(args, model)


if __name__ == "__main__":
    main(args)
