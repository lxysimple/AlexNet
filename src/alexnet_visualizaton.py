# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_visualization.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-14
# @brief      : alexnet_visualization
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    log_dir = os.path.join(BASE_DIR, "..", "results")
    # ----------------------------------- kernel visualization -----------------------------------
    #构建一个写入器，将要展示的数据写入文件
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")

    # m1
    # alexnet = models.alexnet(pretrained=True)

    # m2
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    alexnet = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    alexnet.load_state_dict(pretrained_state_dict)

    kernel_num = -1
    vis_max = 1#可视化卷积层最大到1，即0、1
    for sub_module in alexnet.modules():#遍历所有网络层
        if not isinstance(sub_module, nn.Conv2d):#如果不是卷积层，则跳过
            continue
        kernel_num += 1
        if kernel_num > vis_max:
            break

        kernels = sub_module.weight
        c_out, c_int, k_h, k_w = tuple(kernels.shape)

        # 拆分64个channel
        for o_idx in range(c_out):
            #unsqueeze(1)：在扩展一个维度，在下标为1的地方
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # 获得(3, h, w), 但是make_grid需要 BCHW，这里拓展C维度变为（3， 1， h, w）
            #将多张图片汇总到一张图片上显示
            #kernel_idx：(batchsize=展示图片个数,C,H,W)
            #normalize=True，将像素转换到0~1区间内，可视化时，会自动×255从而映射到0~255区间
            #scale_each=True,是以每一张图片为单位标准化的,单张图片找最值，映射到0~255
            #nrow,展示图占几行
            #是一个一行3列的网格图,kernel_idx=(3,1,h,w)
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
            writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

        #获取make_grid所需要的(B,C,H,W)
        kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
        writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=620)

        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    # ----------------------------------- feature map visualization -----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    # 数据
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")  # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 模型
    # alexnet = models.alexnet(pretrained=True)

    # forward
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理，将fmap_1格式(C,B,H,W)转换为(B,C,H,W)
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=620)
    writer.close()
