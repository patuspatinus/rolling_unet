import torch
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def iou_score(output, target, thresh=0.5, eps=1e-5, ignore_empty=False):
    # output: logits (B, C, H, W)
    # target: (B, C, H, W) với mask 0/1 hoặc 0/255
    prob = torch.sigmoid(output)
    pred = (prob > thresh).float()
    tgt  = (target > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    tgt  = tgt.view(tgt.size(0), -1)

    inter = (pred * tgt).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    tgt_sum  = tgt.sum(dim=1)
    union = pred_sum + tgt_sum - inter

    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred_sum + tgt_sum + eps)

    if ignore_empty:
        keep = (tgt_sum > 0)          # chỉ ảnh có lesion
        iou = iou[keep]
        dice = dice[keep]

    return iou.mean().item(), dice.mean().item()

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
