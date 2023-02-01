import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    # intersecion = np.multiply(output_, target_)
    # # 两者相加，值大于0的部分为交集
    # union = np.asarray(output_ + target_ > 0, np.float32)
    # iou = intersecion.sum() / (union.sum() + 1e-10)

    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def sensitivity(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    intersection = (output * target).sum()
 
    return (intersection + smooth) / \
        (target.sum() + smooth)
        

def precision(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    intersection = (output * target).sum()
 
    return (intersection + smooth) / \
        (output.sum() + smooth)
        