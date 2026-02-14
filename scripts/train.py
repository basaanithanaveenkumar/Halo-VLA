"""
Simple training script for Halo-VLA.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --batch_size 4 --lr 1e-4
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so imports work
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from models.halo_vla import HaloVLM
from dataloader.eo_dataset import build_eo_dataloader
from utils import log_module_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def compute_language_loss(logits, labels, num_prepended):
    """
    Cross-entropy loss on the text portion of the sequence.

    Args:
        logits:        [B, total_len, vocab_size]
        labels:        [B, seq_len]       (−100 for masked positions)
        num_prepended: int — number of image-patch tokens prepended
    """
    # The logits for text tokens start after the prepended image patches.
    # Shift by 1 for next-token prediction: predict token t+1 from position t.
    text_logits = logits[:, num_prepended:-1, :]  # [B, seq_len-1, V]
    target = labels[:, 1:]                         # [B, seq_len-1]

    # Truncate to the shorter of the two (in case of length mismatch)
    min_len = min(text_logits.size(1), target.size(1))
    text_logits = text_logits[:, :min_len, :]
    target = target[:, :min_len]

    loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        target.reshape(-1),
        ignore_index=-100,
    )
    return loss


def compute_action_loss(action_preds, action_targets, action_mask):
    """
    MSE loss between predicted and ground-truth actions.

    Args:
        action_preds:   [B, n_act, chunk_size, action_dim] or None
        action_targets: [B, max_T, action_dim]
        action_mask:    [B, max_T]  (1 for real, 0 for pad)
    """
    if action_preds is None:
        return torch.tensor(0.0)

    # Flatten chunk dimension: [B, n_act * chunk_size, action_dim]
    B, n_act, chunk, dim = action_preds.shape
    preds_flat = action_preds.view(B, n_act * chunk, dim)

    # Align lengths (targets may be longer/shorter)
    T = min(preds_flat.size(1), action_targets.size(1))
    preds_flat = preds_flat[:, :T, :]
    targets = action_targets[:, :T, :]
    mask = action_mask[:, :T].unsqueeze(-1).float()  # [B, T, 1]

    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds_flat.device)

    loss = ((preds_flat - targets) ** 2 * mask).sum() / mask.sum() / dim
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ---- Config ----
    config = HaloVLMConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_chunk_size=args.action_chunk_size,
    )

    # ---- Model ----
    model = HaloVLM(config=config).to(device)
    log_module_parameters(model, model_name="HaloVLM", logger_fn=logger)

    # ---- Dataloader ----
    train_loader = build_eo_dataloader(
        subset=args.subset,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=config.img_size,
        max_seq_len=args.max_seq_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        shuffle=True,
    )
    logger.info("Dataset size: %d  |  Batches: %d", len(train_loader.dataset), len(train_loader))

    # ---- Optimizer & scheduler ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training ----
    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_lang_loss = 0.0
        epoch_act_loss = 0.0
        epoch_total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            images = batch["images"].to(device)              # [B, N, 3, H, W]
            input_ids = batch["input_ids"].to(device)        # [B, seq_len]
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)              # [B, seq_len]
            actions = batch["actions"].to(device)            # [B, T, action_dim]
            action_mask = batch["action_mask"].to(device)    # [B, T]
            states = batch["states"].to(device)              # [B, S, state_dim]

            # Forward
            logits, action_preds = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                states=states,
            )

            # Number of prepended image-patch tokens (for loss alignment)
            num_images = (input_ids == config.image_token_id).sum(dim=1).max().item()
            num_patches = (config.img_size // config.patch_size) ** 2
            num_prepended = num_images * num_patches

            # Losses
            lang_loss = compute_language_loss(logits, labels, num_prepended)
            act_loss = compute_action_loss(action_preds, actions, action_mask)
            total_loss = lang_loss + args.action_loss_weight * act_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_lang_loss += lang_loss.item()
            epoch_act_loss += act_loss.item()
            epoch_total_loss += total_loss.item()

            # Logging
            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d | Step %d/%d | lang=%.4f  act=%.4f  total=%.4f | lr=%.2e",
                    epoch, step, len(train_loader),
                    lang_loss.item(), act_loss.item(), total_loss.item(), lr,
                )

        # End of epoch summary
        n = len(train_loader)
        elapsed = time.time() - t0
        logger.info(
            "=== Epoch %d done in %.1fs | avg lang=%.4f  act=%.4f  total=%.4f ===",
            epoch, elapsed,
            epoch_lang_loss / n, epoch_act_loss / n, epoch_total_loss / n,
        )

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"halo_vla_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint → %s", ckpt_path)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Halo-VLA")

    # Data
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=1)

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--action_loss_weight", type=float, default=1.0)

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging & checkpoints
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--ckpt_dir", default="checkpoints")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
