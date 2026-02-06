import argparse
import os
from glob import glob
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from tqdm import tqdm

import archs
from dataset import Dataset
from utils import AverageMeter
from metrics import iou_score  # khuyên dùng iou_score macro bạn đã sửa (iou, dice, hd95)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='experiment name (folder in models/)')
    p.add_argument('--data_root', required=True, help='dataset root, ví dụ Breast_AD_256')
    p.add_argument('--split', default='val', choices=['train','valid','val','test'])
    p.add_argument('--save_pred', action='store_true', help='save predicted masks to outputs/')
    return p.parse_args()

def seed_torch(seed=1029):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def list_ids(img_dir, img_ext):
    paths = sorted(glob(os.path.join(img_dir, '*' + img_ext)))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

def main():
    seed_torch()
    args = parse_args()

    # ---- load config from training ----
    cfg_path = os.path.join('models', args.name, 'config.yml')
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for k in config.keys():
        print(f'{k}: {config[k]}')
    print('-' * 20)

    cudnn.benchmark = True

    # ---- create model ----
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision']
    ).cuda()

    ckpt_path = os.path.join('models', args.name, 'model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    model.eval()

    # ---- build valid dataset ----
    split = 'val' if args.split == 'val' else args.split

    img_dir = os.path.join(args.data_root, split, 'imgs')
    mask_dir = os.path.join(args.data_root, split, 'masks')  # Dataset sẽ đọc masks/0/...

    img_ids = list_ids(img_dir, config['img_ext'])
    print(f'[{split}] num images = {len(img_ids)}  (img_dir={img_dir})')

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    ds = Dataset(
        img_ids=img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    # ---- meters ----
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    hd95_meter = AverageMeter()

    # output folder
    if args.save_pred:
        out_dir = os.path.join('outputs', args.name, split, '0')
        os.makedirs(out_dir, exist_ok=True)

    # ---- eval loop ----
    with torch.no_grad():
        for inp, tgt, meta in tqdm(loader, total=len(loader)):
            inp = inp.cuda()
            tgt = tgt.cuda()

            logits = model(inp)
            iou, dice = iou_score(logits, tgt)

            iou_meter.update(iou, inp.size(0))
            dice_meter.update(dice, inp.size(0))
            # hd95 có thể nan -> chỉ update nếu là số
            # if hd95 == hd95:  # not nan
            #     hd95_meter.update(hd95, inp.size(0))

            if args.save_pred:
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                pred = (prob >= 0.5).astype(np.uint8) * 255  # (B,1,H,W)

                for i in range(pred.shape[0]):
                    img_id = meta['img_id'][i]
                    cv2.imwrite(os.path.join(out_dir, img_id + '.png'), pred[i, 0])

    print(f'[{split}] IoU  : {iou_meter.avg:.4f}')
    print(f'[{split}] Dice : {dice_meter.avg:.4f}')
    if hd95_meter.count > 0:
        print(f'[{split}] Hd95 : {hd95_meter.avg:.4f}')
    else:
        print(f'[{split}] Hd95 : N/A (nan for all samples)')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
