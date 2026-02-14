"""
EO-Data1.5M Dataloader for Halo-VLA.

Supports the IPEC-COMMUNITY/EO-Data1.5M dataset from HuggingFace,
including all 17 subsets (5 interleaved + 12 QA).

Dataset schema per sample:
    source       : str   — source dataset identifier
    conversation : list  — multi-turn dicts [{"from": ..., "value": ...}, ...]
    image        : PIL.Image or list[PIL.Image] — visual observations
    action       : list/ndarray or None — robot action chunks (continuous)
    state        : list/ndarray or None — robot proprioceptive state

Reference: https://huggingface.co/datasets/IPEC-COMMUNITY/EO-Data1.5M
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image token placeholder used inside conversation text
# ---------------------------------------------------------------------------
IMAGE_TOKEN = "<image>"
ACTION_TOKEN = "<action>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# ---------------------------------------------------------------------------
# All available subsets
# ---------------------------------------------------------------------------
INTERLEAVE_SUBSETS = [
    "interleave-free_chat",
    "interleave-random_qa",
    "interleave-temporal",
    "interleave-trajectory",
    "interleave-video_caption",
]

QA_SUBSETS = [
    "qa-affordance_qa",
    "qa-episode_caption",
    "qa-failure_detection",
    "qa-multiview_qa",
    "qa-object_referring_qa",
    "qa-physical_common_sense",
    "qa-points_qa",
    "qa-process_verification",
    "qa-relation_reasoning",
    "qa-subtask_qa",
    "qa-task_planning",
    "qa-trajectory_qa",
]

ALL_SUBSETS = INTERLEAVE_SUBSETS + QA_SUBSETS


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class EODatasetConfig:
    """Configuration for the EO-Dataset dataloader."""

    dataset_name: str = "IPEC-COMMUNITY/EO-Data1.5M"
    subset: str = "interleave-temporal"
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 512
    max_action_len: int = 64
    action_dim: int = 7  # typical robot action dim (6-DOF + gripper)
    state_dim: int = 7
    max_images: int = 8  # max images per sample for interleaved data
    tokenizer_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    streaming: bool = False
    cache_dir: Optional[str] = None
    # Image normalisation (ImageNet defaults)
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    # Conversation formatting
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"
    # Special token IDs (set after tokenizer init)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    image_token_id: int = 3
    action_token_id: int = 4


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
def build_image_transform(img_size: int, mean: Tuple, std: Tuple):
    """Build torchvision transforms for images."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------
class EODataset(Dataset):
    """
    PyTorch Dataset for IPEC-COMMUNITY/EO-Data1.5M.

    Each __getitem__ returns a dict with keys:
        - images        : Tensor [N, 3, H, W]  (N = number of images, 1..max_images)
        - input_ids     : Tensor [seq_len]       tokenised conversation
        - attention_mask : Tensor [seq_len]
        - labels        : Tensor [seq_len]       shifted targets (-100 for ignored)
        - actions       : Tensor [T, action_dim] continuous action chunks
        - action_mask   : Tensor [T]             1 where action is valid
        - states        : Tensor [T, state_dim]  proprioceptive state
        - state_mask    : Tensor [T]
        - num_images    : int                    actual number of images
        - source        : str                    source dataset id
    """

    def __init__(
        self,
        config: Optional[EODatasetConfig] = None,
        hf_dataset=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = config or EODatasetConfig(**kwargs)
        self._setup_tokenizer(tokenizer)
        self._setup_transform()
        self._load_dataset(hf_dataset)

    # ------------------------------------------------------------------ init
    def _setup_tokenizer(self, tokenizer):
        """Initialise or configure the tokenizer."""
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                trust_remote_code=True,
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens for image / action placeholders if missing
        special_tokens: Dict[str, str] = {}
        additional_special: List[str] = []
        for tok in [IMAGE_TOKEN, ACTION_TOKEN]:
            if tok not in self.tokenizer.get_vocab():
                additional_special.append(tok)
        if additional_special:
            special_tokens["additional_special_tokens"] = additional_special
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)

        # Cache token ids
        self.cfg.pad_token_id = self.tokenizer.pad_token_id
        self.cfg.bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.pad_token_id
        )
        self.cfg.eos_token_id = self.tokenizer.eos_token_id
        self.cfg.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.cfg.action_token_id = self.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)

    def _setup_transform(self):
        self.img_transform = build_image_transform(
            self.cfg.img_size, self.cfg.img_mean, self.cfg.img_std
        )

    def _load_dataset(self, hf_dataset):
        """Load the HuggingFace dataset (or accept a pre-loaded one)."""
        if hf_dataset is not None:
            self.dataset = hf_dataset
            return

        from datasets import load_dataset

        logger.info(
            "Loading %s / %s [%s] …",
            self.cfg.dataset_name,
            self.cfg.subset,
            self.cfg.split,
        )
        self.dataset = load_dataset(
            self.cfg.dataset_name,
            name=self.cfg.subset,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
            cache_dir=self.cfg.cache_dir,
        )
        logger.info("Dataset loaded: %d samples", len(self.dataset))

    # ---------------------------------------------------------- length / get
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]

        # ----- Images -----
        images = self._process_images(sample)

        # ----- Conversation → token ids -----
        input_ids, attention_mask, labels = self._process_conversation(sample)

        # ----- Actions -----
        actions, action_mask = self._process_actions(sample)

        # ----- States -----
        states, state_mask = self._process_states(sample)

        return {
            "images": images,                 # [N, 3, H, W]
            "input_ids": input_ids,           # [seq_len]
            "attention_mask": attention_mask,  # [seq_len]
            "labels": labels,                 # [seq_len]
            "actions": actions,               # [T, action_dim]
            "action_mask": action_mask,        # [T]
            "states": states,                 # [T, state_dim]
            "state_mask": state_mask,          # [T]
            "num_images": images.size(0),
            "source": sample.get("source", ""),
        }

    # ------------------------------------------------------------- images
    def _process_images(self, sample: Dict) -> torch.Tensor:
        """
        Extract and transform images.

        The `image` field can be:
            - a single PIL Image
            - a list of PIL Images (multi-view / temporal)
            - bytes that can be decoded
        Returns Tensor [N, 3, H, W], padded to max_images with zeros.
        """
        from PIL import Image
        import io

        raw = sample.get("image")
        imgs: List[torch.Tensor] = []

        if raw is None:
            # No image → return a black placeholder
            return torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)

        # Normalise to a list
        if not isinstance(raw, (list, tuple)):
            raw = [raw]

        for item in raw[: self.cfg.max_images]:
            if isinstance(item, Image.Image):
                img = item.convert("RGB")
            elif isinstance(item, bytes):
                img = Image.open(io.BytesIO(item)).convert("RGB")
            elif isinstance(item, dict) and "bytes" in item:
                img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
            elif isinstance(item, str):
                img = Image.open(item).convert("RGB")
            else:
                # Try PIL conversion as fallback
                try:
                    img = Image.fromarray(np.uint8(item)).convert("RGB")
                except Exception:
                    logger.warning("Skipping unrecognised image format: %s", type(item))
                    continue
            imgs.append(self.img_transform(img))

        if not imgs:
            return torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)

        return torch.stack(imgs, dim=0)  # [N, 3, H, W]

    # ------------------------------------------------------- conversation
    def _process_conversation(
        self, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flatten multi-turn conversation into a single token sequence.

        Conversation format (HF):
            [
                {"from": "human", "value": "<image>\nWhat is happening?"},
                {"from": "gpt",   "value": "The robot is picking up ..."},
                ...
            ]

        Returns:
            input_ids      : LongTensor [max_seq_len]
            attention_mask  : LongTensor [max_seq_len]
            labels          : LongTensor [max_seq_len]  (-100 for human turns)
        """
        conversation = sample.get("conversation", [])
        if not conversation:
            return self._empty_text()

        text_parts: List[str] = []
        role_is_assistant: List[bool] = []

        for turn in conversation:
            role = turn.get("from", turn.get("role", "human")).lower()
            value = turn.get("value", turn.get("content", ""))

            if role in ("human", "user"):
                text_parts.append(f"{self.cfg.human_prefix}{value}{self.cfg.turn_end}")
                role_is_assistant.append(False)
            else:
                text_parts.append(f"{self.cfg.assistant_prefix}{value}{self.cfg.turn_end}")
                role_is_assistant.append(True)

        full_text = "".join(text_parts)

        # Tokenise
        encoding = self.tokenizer(
            full_text,
            max_length=self.cfg.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)             # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)    # [seq_len]

        # Build labels: mask out human turns (set to -100 so loss ignores them)
        labels = input_ids.clone()
        # Simple heuristic: tokenise each turn individually to find boundaries
        labels = self._mask_human_turns(
            full_text, text_parts, role_is_assistant, labels
        )
        # Also mask padding
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels

    def _mask_human_turns(
        self,
        full_text: str,
        text_parts: List[str],
        role_is_assistant: List[bool],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Set labels to -100 for tokens corresponding to human turns."""
        cursor = 0
        for part, is_assistant in zip(text_parts, role_is_assistant):
            start_char = full_text.find(part, cursor)
            if start_char == -1:
                continue
            end_char = start_char + len(part)

            if not is_assistant:
                # Find token offsets for this span
                prefix = full_text[:end_char]
                prefix_ids = self.tokenizer(
                    prefix,
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                start_prefix = full_text[:start_char]
                start_ids = self.tokenizer(
                    start_prefix,
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                tok_start = len(start_ids)
                tok_end = len(prefix_ids)
                if tok_end <= self.cfg.max_seq_len:
                    labels[tok_start:tok_end] = -100

            cursor = end_char
        return labels

    def _empty_text(self):
        """Return padded empty tensors when no conversation is present."""
        ids = torch.full((self.cfg.max_seq_len,), self.cfg.pad_token_id, dtype=torch.long)
        mask = torch.zeros(self.cfg.max_seq_len, dtype=torch.long)
        labels = torch.full((self.cfg.max_seq_len,), -100, dtype=torch.long)
        return ids, mask, labels

    # --------------------------------------------------------- actions
    def _process_actions(
        self, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process robot action chunks into a padded tensor.

        Actions are continuous values, typically 7-dim (6-DOF + gripper).
        Returns:
            actions     : FloatTensor [max_action_len, action_dim]
            action_mask : LongTensor  [max_action_len]
        """
        raw = sample.get("action")

        actions = torch.zeros(self.cfg.max_action_len, self.cfg.action_dim)
        action_mask = torch.zeros(self.cfg.max_action_len, dtype=torch.long)

        if raw is None:
            return actions, action_mask

        arr = self._to_numpy(raw)
        if arr is None or arr.size == 0:
            return actions, action_mask

        # Support 1-D (single step) or 2-D (chunk)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        T = min(arr.shape[0], self.cfg.max_action_len)
        D = min(arr.shape[1], self.cfg.action_dim)

        actions[:T, :D] = torch.tensor(arr[:T, :D], dtype=torch.float32)
        action_mask[:T] = 1

        return actions, action_mask

    # ---------------------------------------------------------- states
    def _process_states(
        self, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process robot proprioceptive state."""
        raw = sample.get("state")

        states = torch.zeros(self.cfg.max_action_len, self.cfg.state_dim)
        state_mask = torch.zeros(self.cfg.max_action_len, dtype=torch.long)

        if raw is None:
            return states, state_mask

        arr = self._to_numpy(raw)
        if arr is None or arr.size == 0:
            return states, state_mask

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        T = min(arr.shape[0], self.cfg.max_action_len)
        D = min(arr.shape[1], self.cfg.state_dim)

        states[:T, :D] = torch.tensor(arr[:T, :D], dtype=torch.float32)
        state_mask[:T] = 1

        return states, state_mask

    # ---------------------------------------------------------- utils
    @staticmethod
    def _to_numpy(raw) -> Optional[np.ndarray]:
        """Convert various action/state representations to numpy."""
        if isinstance(raw, np.ndarray):
            return raw.astype(np.float32)
        if isinstance(raw, torch.Tensor):
            return raw.numpy().astype(np.float32)
        if isinstance(raw, (list, tuple)):
            try:
                return np.array(raw, dtype=np.float32)
            except (ValueError, TypeError):
                return None
        return None


# ---------------------------------------------------------------------------
# Collate function for variable-length image sequences
# ---------------------------------------------------------------------------
def eo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that pads images to the same count per batch and
    stacks everything else normally.

    Returns:
        images        : [B, max_N, 3, H, W]
        image_mask    : [B, max_N]            (1 for real images, 0 for pad)
        input_ids     : [B, seq_len]
        attention_mask : [B, seq_len]
        labels        : [B, seq_len]
        actions       : [B, T, action_dim]
        action_mask   : [B, T]
        states        : [B, T, state_dim]
        state_mask    : [B, T]
    """
    max_n_images = max(b["num_images"] for b in batch)
    C = batch[0]["images"].size(1)
    H = batch[0]["images"].size(2)
    W = batch[0]["images"].size(3) if batch[0]["images"].dim() == 4 else batch[0]["images"].size(2)

    B = len(batch)
    padded_images = torch.zeros(B, max_n_images, C, H, W)
    image_mask = torch.zeros(B, max_n_images, dtype=torch.long)

    input_ids = []
    attention_masks = []
    labels_list = []
    actions = []
    action_masks = []
    states = []
    state_masks = []
    sources = []

    for i, b in enumerate(batch):
        n = b["num_images"]
        padded_images[i, :n] = b["images"][:n]
        image_mask[i, :n] = 1

        input_ids.append(b["input_ids"])
        attention_masks.append(b["attention_mask"])
        labels_list.append(b["labels"])
        actions.append(b["actions"])
        action_masks.append(b["action_mask"])
        states.append(b["states"])
        state_masks.append(b["state_mask"])
        sources.append(b["source"])

    return {
        "images": padded_images,                          # [B, max_N, 3, H, W]
        "image_mask": image_mask,                         # [B, max_N]
        "input_ids": torch.stack(input_ids),              # [B, seq_len]
        "attention_mask": torch.stack(attention_masks),    # [B, seq_len]
        "labels": torch.stack(labels_list),               # [B, seq_len]
        "actions": torch.stack(actions),                  # [B, T, action_dim]
        "action_mask": torch.stack(action_masks),         # [B, T]
        "states": torch.stack(states),                    # [B, T, state_dim]
        "state_mask": torch.stack(state_masks),           # [B, T]
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_eo_dataloader(
    subset: str = "interleave-temporal",
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    tokenizer=None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 64,
    action_dim: int = 7,
    state_dim: int = 7,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a ready-to-use DataLoader for EO-Data1.5M.

    Example usage::

        from data import build_eo_dataloader

        loader = build_eo_dataloader(
            subset="interleave-temporal",
            batch_size=8,
            num_workers=4,
        )
        for batch in loader:
            images = batch["images"]           # [B, N, 3, 224, 224]
            input_ids = batch["input_ids"]     # [B, 512]
            actions = batch["actions"]         # [B, 64, 7]
            ...
    """
    cfg = EODatasetConfig(
        subset=subset,
        split=split,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        streaming=streaming,
        cache_dir=cache_dir,
    )

    dataset = EODataset(config=cfg, tokenizer=tokenizer, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not streaming,
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EO-Data1.5M dataloader test")
    parser.add_argument("--subset", default="interleave-temporal")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    loader = build_eo_dataloader(
        subset=args.subset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"\nDataset size: {len(loader.dataset)}")
    print(f"Batches:      {len(loader)}\n")

    batch = next(iter(loader))

    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s} → {val.shape}  dtype={val.dtype}")
        else:
            print(f"  {key:20s} → {type(val).__name__}: {val}")

    print("\n✓ Dataloader works correctly.")
