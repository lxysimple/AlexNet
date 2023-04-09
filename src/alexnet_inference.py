# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-13
# @brief      : inference demo
"""

import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import time
import json
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#如果电脑中有GPU，则device=gpu，否则device=cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#解决问题:torch包中包含了名为libiomp5md.dll的文件，与Anaconda环境中的同一个文件出现了某种冲突，所以需要删除一个
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()#掉包侠，进入该函数，再进入一个函数，就有实现细节
    pretrained_state_dict = torch.load(path_state_dict)#将预训练模型状态装入gpu
    model.load_state_dict(pretrained_state_dict)#将本模型状态改变成储存的状态
    model.eval()#将模型状态置为测试状态，不会dropout，结果要×2

    if vis_model:
        from torchsummary import summary
        #这个summary将输入一个虚拟的数据，去测试模型各层输出尺寸和参数总量，并输出返回，非常实用
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


def process_img(path_img):

    # hard code
    #是从imageNet中统计出所有图片像素的均值、标准差
    #是把0~255/255后转化到0~1区间内统计出来的
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    #创建一个格式转换对象
    inference_transform = transforms.Compose([
        transforms.Resize(256),#缩放到256
        transforms.CenterCrop((224, 224)),#从中心裁剪(224, 224)
        transforms.ToTensor(),#转化成张量，并且[0,255]/255=>[0,1]
        transforms.Normalize(norm_mean, norm_std),#减均值，除方差
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


if __name__ == "__main__":

    # config
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    # path_img = os.path.join(BASE_DIR, "..", "data", "Golden Retriever from baidu.jpg")
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")#测试图位置
    #以下是1000个类别映射关系所保存的文件
    path_classnames = os.path.join(BASE_DIR, "..", "data", "imagenet1000.json")
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "imagenet_classnames.txt")

    # load class names
    #ctrl+左键点击=跳转到该函数
    #加载txt和json文件
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    #True表示也会尝试去网络上加载模型参数
    alexnet_model = get_model(path_state_dict, True)

    # 3/5 inference  tensor --> vector 4D->1D
    with torch.no_grad():
        time_tic = time.time()
        outputs = alexnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)#找出最大概率那个类别
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)#找出前5大概率的类别

    #在cpu上根据id找对应的名词和图片位置，输出英文版和中文版
    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15+idx*30, "top {}:{}".format(idx+1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()

