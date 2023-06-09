# -*- coding: utf-8 -*-
"""
# @file name  : my_dataset.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-02-14
# @brief      : 数据集Dataset定义
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    #一般该函数都是固定存在的
    #根据一个索引，从内存中返回图片+图片的类别
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    # 一般该函数都是固定存在的
    #返回当前数据集的长度
    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    #将数据载入内存
    #将图片按某种算法分成1+9份，当mode=train时，返回9份训练数据，当mode=valid，返回1份数据集测试
    def _get_img_info(self):

        #获取该路径下所有文件夹名
        img_names = os.listdir(self.data_dir)
        #获取img_names各文件夹下所有.jpg图片文件
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        random.seed(self.rng_seed)
        random.shuffle(img_names)

        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]

        split_idx = int(len(img_labels) * self.split_n)  # 25000* 0.9 = 22500
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]     # 数据集90%训练
            # img_set = img_names[:22500]     #  hard code 数据集90%训练
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        #将每个图片名字前+上其路径，转化为绝对路径
        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        #每个绝对路径都配上对应的标签
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info
