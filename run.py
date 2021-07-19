from PIL.Image import Image
from 网络模型文件 import 网络模型
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models 
#import visdom
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = "main")

#-------------------------------- program parameter -------------------------------------#
parser.add_argument('--network', type = str, \
    default = 'lenet+minst', help = 'network and dataset')
parser.add_argument('--batch_size', type = int, \
    default = 512, help = 'test batch size')
parser.add_argument('--model', type = str, \
    default = './lenet_minst.pth', help = 'model location')
parser.add_argument('--epoch', type = str, \
    default = 16, help = 'epoch number')

args = parser.parse_args()
#----------------------------------------------------------------------------------------#

# 网络模型初始化函数
def initialNetwork() :
    print("network and dataset: ", args.network)
    if args.network == 
        cnn = 网络例化
        cnn = load_model(cnn, args.model)

    return cnn

# 训练集和测试集的加载函数定义
def getTrainDataset() :
    if 'dataset_name' in args.network:
        train_dataset = 
    return train_dataset

def getTestDataset() :
    if 'dataset_name' in args.network:
        test_dataset = 
    return test_dataset

def train(cnn, train_dataset):

    for i, (data, target) in enumerate(train_dataset) :       
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        # 推理
        output = cnn(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 

def test(cnn, test_dataset):

    total_correct = 0
    avg_loss = 0.0

    for i, (data, target) in enumerate(test_dataset):
        output = cnn(data)
        avg_loss += criterion(output, target).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(target.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def main():
    # 加载
    cnn = initialNetwork()
    train_dataset = getTrainDataset()
    test_dataset = getTestDataset()

    # 定义损失函数
    criterion = 
    # 定义优化器
    optimizer = 

    # 训练
    for epoch in range(args.epoch):
        train(cnn, train_dataset)

    # 测试
    test()
    # 保存模型
    torch.save(参数)

if __name__ == '__main__':
    main()