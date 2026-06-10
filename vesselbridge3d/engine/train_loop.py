"""Training / evaluation loop (model-agnostic via the model registry)."""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..common.constants import (
    DEFAULT_DECODER_BASE_CHANNELS,
    DEFAULT_DECODER_UP_FACTOR,
    DEFAULT_MODEL_TYPE,
)
from ..common.utils import colorize, get_logger, make_palette, set_seed
from ..data import (
    VolumeDataset3D,
    compute_class_histogram,
    init_prior_bias_for_head,
    make_class_weights,
    make_collate_pad_3d,
)
from ..models import build_model
from .losses import DC_and_CE_loss

logger = get_logger()


def _resize_logits_to_target(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 5: # [B, K, D, H, W]
        if logits.shape[-3:] != y.shape[-3:]:
            logits = F.interpolate(logits, size=y.shape[-3:], mode='trilinear', align_corners=False)
    else: # [B, K, H, W]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
    return logits


# Eval (returns CE, Dice, Total averages)
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_ce = total_dice = total_total = total_topo = 0.0
    count = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, m = batch
            m = m.to(device)
        else:
            x, y = batch
            m = None
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = _resize_logits_to_target(logits, y)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

        out = criterion(logits, y, mask=m)
        if isinstance(out, tuple) and len(out) == 4:
            ce_loss, dice_loss, topo_loss, total_loss = out
        else:
            ce_loss, dice_loss, total_loss = out
            topo_loss = torch.tensor(0.0, device=logits.device)

        bs = x.size(0)
        total_ce   += float(ce_loss.item())   * bs
        total_dice += float(dice_loss.item()) * bs
        total_topo += float(topo_loss.item()) * bs
        total_total+= float(total_loss.item())* bs
        count += bs

    return total_ce / count, total_dice / count, total_topo / count, total_total / count


# Training (with train_list / val_list)
def train(train_list: str,
          val_list: str,
          num_classes: int,
          epochs: int = 50,
          batch_size: int = 16,
          accumulation_steps: int = 8,
          lr: float = 1e-4,
          min_lr: float = 1e-5,
          drop_empty: bool = True,
          seed: int = 42,
          bg_weight: float = 0.05,
          use_mfb: bool = False,
          min_fg_frac: float = 0.002,
          use_topk: bool = False,
          topk_percent: float = 10.0,
          init_prior_bias: bool = False,
          img_size: Tuple[int, int] = (224, 224),
          use_cldice: bool = False, cldice_weight: float = 0.1, cldice_iters: int = 20,
          use_skelrecall: bool = False, skelrecall_weight: float = 0.1, skelrecall_iters: int = 20,
          aug: bool = False,
          aug_rotate: float = 15.0,
          aug_scale_min: float = 0.90, aug_scale_max: float = 1.10,
          aug_translate: float = 0.10,
          aug_hflip_p: float = 0.0, aug_vflip_p: float = 0.0,
          aug_noise_p: float = 0.2, aug_noise_std: float = 0.01,
          aug_bc_p: float = 0.3, aug_brightness: float = 0.10, aug_contrast: float = 0.10,
          use_25d: bool = False,
          z_stack: int = 5,
          fuse_mode: str = 'conv3d-center',
          use_3d: bool = False,
          depth_min_fg_frac: float = 0.0,
          crop_depth: int = 64,
          save_dir="/path/to/save_dir",
          # --- threaded explicitly (previously read from a module-global args) ---
          log_dir: str = "/path/to/log_dir",
          model_type: str = DEFAULT_MODEL_TYPE,
          vit_layers: Tuple[int, ...] = (2, 5, 8, 11),
          decoder_up_factor: int = DEFAULT_DECODER_UP_FACTOR,
          vit_chunk_slices: int = 8,
          vit_amp: bool = True,
          config_snapshot: dict | None = None):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_snapshot = config_snapshot if config_snapshot is not None else {}

    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(log_dir, f"train_2d_refine_head_{run_id}.log")
    logger.info(f"[INFO] Logging to {log_file}")
    logger.info(f"[INFO] Saving weights to existing directory: {save_dir}")

    # header
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write(f"# train_list={train_list}\n# val_list={val_list}\n")
        f.write(f"# num_classes={num_classes} epochs={epochs} batch_size={batch_size} "
                f"lr={lr} min_lr={min_lr} img_size={img_size} use_3d={use_3d} "
                f"depth_min_fg_frac={depth_min_fg_frac} crop_depth={crop_depth}\n")
        f.write(f"# save_dir={save_dir}\n")

    assert not (use_25d and use_3d), "use_25d and use_3d cannot be used at the same time!"

    # Datasets / Loaders
    ds_train = VolumeDataset3D(train_list, out_size=img_size,
                                drop_empty=drop_empty, min_fg_frac=min_fg_frac,
                                augment=aug,
                                hflip_p=aug_hflip_p, vflip_p=aug_vflip_p,
                                rotate_deg=aug_rotate, scale_min=aug_scale_min, scale_max=aug_scale_max,
                                translate_frac=aug_translate,
                                noise_p=aug_noise_p, noise_std=aug_noise_std,
                                bc_p=aug_bc_p, brightness=aug_brightness, contrast=aug_contrast,
                                crop_depth=crop_depth)
    ds_val = VolumeDataset3D(val_list if val_list is not None else train_list,
                                out_size=img_size, drop_empty=drop_empty, min_fg_frac=min_fg_frac,
                                augment=False, crop_depth=crop_depth)

    collate_fn_train = make_collate_pad_3d(depth_min_fg_frac)
    collate_fn_eval = make_collate_pad_3d(depth_min_fg_frac)

    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn_train
    )
    loader_eval = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn_eval
    )

    # Model
    model = build_model(
                model_type,
                num_classes=num_classes,
                vit_layers=tuple(vit_layers),
                decoder_base_channels=DEFAULT_DECODER_BASE_CHANNELS,
                decoder_up_factor=decoder_up_factor,
                vit_chunk_slices=vit_chunk_slices,
                vit_amp=vit_amp,
                use_se=True
            ).to(device)
    trainable_params = model.trainable_parameters()

    # parameter counting helpers & prints
    def count_params(module: nn.Module, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())

    def fmt(n: int) -> str:
        return f"{n:,}"

    # Granular components
    # Adapter: (a) slice adapter (frozen)  (b) 3D ConvNeXt adapter
    adapter_slice_total     = count_params(model.adapter_slice, trainable_only=False)
    adapter_slice_trainable = count_params(model.adapter_slice, trainable_only=True)

    adapter3d_total     = count_params(model.adapter3d, trainable_only=False)
    adapter3d_trainable = count_params(model.adapter3d, trainable_only=True)

    # Small adapter-related projection/gate layers
    proj_gate_modules = nn.ModuleList([
        model.f2_proj_ref,   # 48 -> 64 (3D)
        model.f3_proj_C,     # 64 -> C  (3D)
        model.f3_gate3d,     # gate conv
        model.f2_hr_reduce3d # for HR 2D head
    ])
    proj_gate_total     = count_params(proj_gate_modules, trainable_only=False)
    proj_gate_trainable = count_params(proj_gate_modules, trainable_only=True)

    # Aggregator (wrapper) and its FFA core
    aggregator_total     = count_params(model.shared_agg, trainable_only=False)
    aggregator_trainable = count_params(model.shared_agg, trainable_only=True)

    ffa_core_total     = count_params(model.shared_agg.core, trainable_only=False)
    ffa_core_trainable = count_params(model.shared_agg.core, trainable_only=True)

    # Segmentation heads: 3D refine head + 2D HR head
    seg_heads = nn.ModuleList([model.head, model.hr2d])
    seg_total     = count_params(seg_heads, trainable_only=False)
    seg_trainable = count_params(seg_heads, trainable_only=True)

    # ViT encoder (frozen, for reference)
    vit_total     = count_params(model.encoder, trainable_only=False)
    vit_trainable = count_params(model.encoder, trainable_only=True)

    # Model totals
    model_total_all     = count_params(model, trainable_only=False)
    model_total_trainable = count_params(model, trainable_only=True)

    logger.info("===== Parameter Counts =====")
    logger.info(f"[Adapter (slice/frozen)] total={fmt(adapter_slice_total)} | trainable={fmt(adapter_slice_trainable)}")
    logger.info(f"[Adapter3D (ConvNeXt)]  total={fmt(adapter3d_total)} | trainable={fmt(adapter3d_trainable)}")
    logger.info(f"[Adapter Proj/Gates]    total={fmt(proj_gate_total)} | trainable={fmt(proj_gate_trainable)}")
    logger.info(f"[Aggregator (wrapper)]  total={fmt(aggregator_total)} | trainable={fmt(aggregator_trainable)}")
    logger.info(f"[FFA core]              total={fmt(ffa_core_total)} | trainable={fmt(ffa_core_trainable)}")
    logger.info(f"[Segmentation heads]    total={fmt(seg_total)} | trainable={fmt(seg_trainable)}")
    logger.info(f"[ViT encoder (frozen)]  total={fmt(vit_total)} | trainable={fmt(vit_trainable)}")
    logger.info("--------------------------------------------")
    logger.info(f"[MODEL TOTAL]           total={fmt(model_total_all)} | trainable={fmt(model_total_trainable)}")
    logger.info("============================================")

    # Class weights & optional prior bias (TRAIN set)
    counts = compute_class_histogram(ds_train, num_classes)
    class_weights = make_class_weights(num_classes, counts, bg_weight=bg_weight, use_mfb=use_mfb)
    if init_prior_bias:
        cls_conv_3d = init_prior_bias_for_head(model.head,  num_classes, counts)
        cls_conv_2d = init_prior_bias_for_head(model.hr2d, num_classes, counts)
        logger.info(f"[INFO] Initialized head bias (3D refine): {cls_conv_3d}")
        logger.info(f"[INFO] Initialized head bias (2D HR): {cls_conv_2d}")

    criterion = DC_and_CE_loss(
        num_classes=num_classes,
        class_weights=class_weights,
        weight_ce=1.0, weight_dice=1.0,
        use_topk=use_topk, topk_percent=topk_percent,
        use_cldice=use_cldice, cldice_weight=cldice_weight, cldice_iters=cldice_iters,
        use_skelrecall=use_skelrecall, skelrecall_weight=skelrecall_weight, skelrecall_iters=skelrecall_iters
    )

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.05)

    def lr_lambda(current_epoch):
        warmup_epochs = 50
        total_epochs = epochs

        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            # Cosine Decay: 1.0 -> 0.0
            progress = float(current_epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # color palette
    palette = make_palette(num_classes)

    start_epoch = 1
    best_loss = float('inf')
    best_dice = float('inf')

    latest_ckpt_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    if os.path.isfile(latest_ckpt_path):
        logger.info(f"[INFO] Found latest checkpoint: {latest_ckpt_path}")
        logger.info("[INFO] Resuming training...")
        loc = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(latest_ckpt_path, map_location=loc)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_dice = checkpoint.get('best_dice', float('inf'))

        logger.info(f"[INFO] Resumed from Epoch {checkpoint['epoch']}. Next Epoch: {start_epoch}")
    else:
        logger.info("[INFO] No latest checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, epochs+1):
        model.train()

        tr_ce = tr_dice = tr_topo = tr_total = 0.0
        count = 0

        for i, batch_item in enumerate(loader):
            if use_3d:
                x, y, m = batch_item
                m = m.to(device, non_blocking=True)
            else:
                x, y = batch_item
                m = None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                logits = _resize_logits_to_target(logits, y)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

            # fp32
            out = criterion(logits.float(), y, mask=m)
            if isinstance(out, tuple) and len(out) == 4:
                ce_loss, dice_loss, topo_loss, total_loss = out
            else:
                ce_loss, dice_loss, total_loss = out
                topo_loss = torch.tensor(0.0, device=logits.device)

            loss_normalized = total_loss / accumulation_steps
            loss_normalized.backward()

            # Max norm 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = x.size(0)
            tr_ce    += float(ce_loss.item())   * bs
            tr_dice  += float(dice_loss.item()) * bs
            tr_topo  += float(topo_loss.item()) * bs
            tr_total += float(total_loss.item()) * bs
            count += bs

        if len(loader) % accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        tr_ce   /= count; tr_dice /= count; tr_topo /= count; tr_total /= count

        ev_ce, ev_dice, ev_topo, ev_total = eval_epoch(model, loader_eval, criterion, device)

        cur_lr = optimizer.param_groups[0]["lr"]
        msg = (f"[Epoch {epoch:03d}] lr={cur_lr:.2e} "
               f"train: CE={tr_ce:.4f}, Dice={tr_dice:.4f}, Topo={tr_topo:.4f}, Total={tr_total:.4f} | "
               f"eval:  CE={ev_ce:.4f}, Dice={ev_dice:.4f}, Topo={ev_topo:.4f}, Total={ev_total:.4f}")
        logger.info(msg)

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except Exception as e:
            logger.info(f"[WARN] Failed to write log: {e}")

        # update LR per epoch
        scheduler.step()

        if epoch % 20 == 0:
            model.eval()
            vis_batch = next(iter(loader_eval))
            if use_3d:
                x_vis, y_vis, m_vis = vis_batch
                m_vis = m_vis.to(device)
            else:
                x_vis, y_vis = vis_batch
                m_vis = None
            x_vis, y_vis = x_vis.to(device), y_vis.to(device)
            with torch.no_grad():
                logits_vis = model(x_vis)
                logits_vis = _resize_logits_to_target(logits_vis, y_vis)
                preds = logits_vis.argmax(1).detach().cpu().numpy()

            if logits_vis.ndim == 4:
                # 2D/2.5D: [B,K,H,W] -> visualize first sample
                gt = y_vis[0].detach().cpu().numpy().astype(np.int64)
                pr = preds[0].astype(np.int64)
            else:
                # 3D: [B,K,D,H,W] -> central slice of first sample
                D = y_vis.shape[1]
                d = D // 2
                gt = y_vis[0, d].detach().cpu().numpy().astype(np.int64)
                pr = preds[0, d].astype(np.int64)

            gt_vis = colorize(gt, palette); pr_vis = colorize(pr, palette)
            concat = np.concatenate([gt_vis, pr_vis], axis=1)
            out_path = os.path.join("/path/to/vis_outputs/", f"epoch{epoch:03d}_sample0.png")
            imageio.imwrite(out_path, concat.astype(np.uint8))
            logger.info(f"[INFO] Saved color visualization to {out_path}")

        if ev_dice < best_dice:
            best_dice = ev_dice
            best_dice_path = os.path.join(save_dir, 'minibaseline_best_dice.pt')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                        best_dice_path)
            logger.info(f"[INFO] Saved checkpoint: minibaseline_best_dice.pt (loss={best_loss:.4f}), dice={best_dice:.4f}")

        if ev_total < best_loss:
            best_loss = ev_total
            best_loss_path = os.path.join(save_dir, 'minibaseline_best_total.pt')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                        best_loss_path)
            logger.info(f"[INFO] Saved checkpoint: minibaseline_best_total.pt (loss={best_loss:.4f}), dice={best_dice:.4f}")

        # Latest Checkpoint (For Resuming) - always save this
        latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_dice': best_dice,
            'args': config_snapshot
        }, latest_path)

    # Save Final Checkpoint
    final_path = os.path.join(save_dir, 'final_checkpoint.pt')
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'final_loss': ev_total,
        'final_dice': ev_dice,
        'args': config_snapshot
    }, final_path)
    logger.info(f"[INFO] Saved FINAL checkpoint to: {final_path}")
