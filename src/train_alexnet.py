# -*- coding: utf-8 -*-
"""
# @file name  : train_alexnet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-14
# @brief      : alexnet traning
"""
#也可适应于微调

import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.models as models
from tools.my_dataset import CatDogDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


if __name__ == "__main__":

    # config
    data_dir = os.path.join(BASE_DIR, "..", "data", "train")
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    num_classes = 2

    # MAX_EPOCH = 3       # 可自行修改
    MAX_EPOCH = 1  # 可自行修改
    BATCH_SIZE = 128    # 可自行修改
    LR = 0.001          # 可自行修改
    log_interval = 1    # 可自行修改
    val_interval = 1    # 可自行修改
    classes = 2
    start_epoch = -1
    lr_decay_step = 1   # 可自行修改

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256)),      # 将短边变为256，长边自动随之改变
        transforms.CenterCrop(256),     #在中间裁剪一个256×256
        transforms.RandomCrop(224),     #随机裁剪224×224，一条边上的可能为(256-224)=32，一共有32×32种可能
        transforms.RandomHorizontalFlip(p=0.5),#每张图片有50%概率水平翻转，则一共有2048种可能
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        #这种裁剪方法获得10×(B,C,H,W)个图片，依次单独拿出来做ToTensor
        transforms.TenCrop(224, vertical_flip=False),

        #crops：装了10×(B,C,H,W)个图片，crop：1张(B,C,H,W)
        #先将一张图片转化成张量，再标准化，最后将10张转化好的图片放到一起
        transforms.Lambda(lambda crops: torch.stack([
            normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])

    # 构建MyDataset实例
    train_data = CatDogDataset(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = CatDogDataset(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================ step 2/5 模型 ============================
    alexnet_model = get_model(path_state_dict, False)

    #得到最后全连接层的输入长度
    num_ftrs = alexnet_model.classifier._modules["6"].in_features
    #创建一个新的全连接层，使其输入保持原样，输出为2分类
    alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)

    alexnet_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    #因为在提取图像特征的前几个卷积层，提取的特征都很基础与普遍，不需要再学习了
    # flag = 0
    flag = 1
    if flag:
        fc_params_id = list(map(id, alexnet_model.classifier.parameters()))  # 返回的是parameters的 内存地址
        #找到模型所有可学习参数中，非全连接层部分
        base_params = filter(lambda p: id(p) not in fc_params_id, alexnet_model.parameters())
        #将卷积层学习率设置为很小，全连接层相对较大，卷积层学习率也可置0，这样就等价于冻结卷积层
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.0},  # 0
            {'params': alexnet_model.classifier.parameters(), 'lr': LR}], momentum=0.9)

    else:
        #momentum=0.9 表示每次更新时，将上一次的更新方向所占的比例设为 0.9，本次更新方向所占的比例设为 0.1
        optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)  # 选择优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)  # 设置学习率下降策略
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=5)

# ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        alexnet_model.train()
        for i, data in enumerate(train_loader):#一次是一批batchsize数据

            # if i > 1:
            #     break

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:#多少个batchsize记录一次均值loss
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            alexnet_model.eval()
            with torch.no_grad():
                #这里一份测试数据被3个epoch训练的模型共用
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    #ncrops：被裁剪的个数
                    bs, ncrops, c, h, w = inputs.size()     # [4, 10, 3, 224, 224]
                    #view()方法来重新组织张量的维度，-1表示将4×10自动转化成40
                    outputs = alexnet_model(inputs.view(-1, c, h, w))
                    #取每个样本多次裁剪最终2分类概率的均值，(bs,mcrops,2)，在mcrops这个维度取均值
                    #得到(bs,1,2)=>(bs,2),即(4,2)
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                    #loss是batchsize个loss
                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val/len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))
            alexnet_model.train()

    #train_curve长度是在所有epoch中，一次取batchsize大小数据的次数
    train_x = range(len(train_curve))
    train_y = train_curve

    #len(valid_curve)×epoch间隔数×一次epoch中取batchsize大小数据的次数==len(train_curve)
    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()





