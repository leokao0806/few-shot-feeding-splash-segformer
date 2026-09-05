import argparse
import multiprocessing as mp
import time
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# 核心組件
from semseg.models.subclass_segformer_unified import SubclassSegFormer_Unified
from dataset import RippleDataset
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.optimizers import get_optimizer
from semseg.schedulers import get_scheduler
from semseg.utils.utils import fix_seeds
from val_unified import evaluate


def main(cfg, k, save_dir):
    start = time.time()
    best_mIoU = 0.0
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']

    dataset_field = dataset_cfg['FIELD'].split(',')

    # --- 歷史數據紀錄表 ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }

    # 1. 資料集準備
    print(f">>> [Stage 1] 基礎模型重訓練: 使用 ImageNet 權重 + RGB 原圖")
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'],
                                            aug=train_cfg['AUG'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = RippleDataset(dataset_cfg['ROOT'], dataset_field, 'train', traintransform, k, sample_num=0)
    valset = RippleDataset(dataset_cfg['ROOT'], dataset_field, 'val', valtransform, k)

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, num_workers=num_workers,
                             drop_last=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=1, num_workers=num_workers, pin_memory=True)

    # 2. 模型初始化
    model = SubclassSegFormer_Unified(
        backbone=model_cfg['BACKBONE'],
        num_classes=2,
        num_prompts=cfg.get('PROMPT_LEARNING', {}).get('NUM_PROMPTS', 5)
    )

    if model_cfg.get('PRETRAINED'):
        print(f"[*] Loading Official ImageNet Weights: {model_cfg['PRETRAINED']}")
        model.init_pretrained(model_cfg['PRETRAINED'])

    model = model.to(device)

    # 凍結策略
    for name, param in model.named_parameters():
        if 'q_prime' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 3. 訓練組件
    optimizer = get_optimizer(model, optim_cfg['NAME'], optim_cfg['LR'], optim_cfg['WEIGHT_DECAY'])
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, train_cfg['EPOCHS'] * iters_per_epoch, sched_cfg['POWER'],
                              iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])

    # 4. 訓練迴圈
    for epoch in range(train_cfg['EPOCHS']):
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch + 1}/{train_cfg['EPOCHS']}]")

        for iter, (img, lbl, _) in pbar:
            img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=train_cfg['AMP']):
                # 強制執行 q_prime=False (路徑 C)
                logits = model(img, q_prime=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_description(f"Epoch: [{epoch + 1}] Loss: {train_loss / (iter + 1):.4f}")

        # --- 計算本輪平均訓練 Loss ---
        avg_train_loss = train_loss / iters_per_epoch
        history['train_loss'].append(avg_train_loss)

        # 5. 驗證與存檔 (此區塊已移出 iter 迴圈，確保每輪 Epoch 執行一次)
        if (epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0:
            acc, macc, f1, mf1, ious, miou, val_loss, _ = evaluate(model, valloader, device, loss_fn)

            ripple_iou = ious[1]
            history['val_loss'].append(val_loss)
            history['val_iou'].append(ripple_iou)

            print(f"\n--- Epoch [{epoch + 1}] Validation ---")
            print(f"Ripple IoU: {ripple_iou:.4f} | mIoU: {miou:.4f} | Val Loss: {val_loss:.4f}")

            if ripple_iou > best_mIoU:
                best_mIoU = ripple_iou
                save_path = save_dir / "base_model_imagenet_L2_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"🏆 New Best Ripple IoU: {best_mIoU:.4f} -> Saved!")

            print("-" * 40)

            # 每個 Eval 週期後存一次 history，防止當機數據丟失
            np.save(save_dir / "training_history.base_npy", history)

    # 最終儲存
    np.save(save_dir / "training_history.base_npy", history)
    torch.save(model.state_dict(), save_dir / "base_model_imagenet_L2_final.pth")

    end = time.time()
    elapsed = end - start
    print(f"✅ Retrain 完成。最佳 IoU: {best_mIoU:.4f} | 總耗時: {elapsed / 3600:.2f} 小時")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ripple_prompt_stage1.yaml')
    parser.add_argument('--k', type=int, default=0)
    args = parser.parse_args()

    with open(args.cfg) as f: config = yaml.load(f, Loader=yaml.SafeLoader)
    fix_seeds(3402)
    save_dir = Path(config['SAVE_DIR']) / f"retrain_imagenet_base_{args.k}"
    save_dir.mkdir(parents=True, exist_ok=True)
    main(config, args.k, save_dir)