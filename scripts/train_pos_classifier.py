"""
Train POS Token Classifier with TiBERT
=======================================
Fine-tunes the TiBERT-classical-spm encoder for token-level POS classification
using ALL 91 raw SegPOS labels (no simplification).

Architecture:
    TiBERT-classical-spm encoder (fine-tuned, last 6 layers)
        ↓  last_hidden_state [batch, seq_len, 768]
    TokenClassificationHead: Linear(768, num_labels=91)
        ↓
    CrossEntropyLoss (token-level, -100 for special/padding tokens)

Case-particle loss weighting: ×2 for case particle labels.
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch.nn.functional as F
from TorchCRF import CRF
from torch.amp import GradScaler, autocast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, BertConfig, BertModel

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR    = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
DATA_DIR     = Path(__file__).parent.parent / "data"  / "corpus" / "pos_dataset"
OUTPUT_DIR   = Path(__file__).parent.parent / "model" / "pos_classifier"
TOKENIZER_DIR = MODEL_DIR

sys.path.insert(0, str(Path(__file__).parent))
from continued_pretrain import ClassicalTibetanTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

class Config:
    # Model
    hidden_size     = 768
    vocab_size       = 32007     # must match TiBERT encoder vocab (actual: 32007)
    encoder_freeze  = False     # unfreeze encoder — full data justifies fine-tuning

    # num_labels: read dynamically from label_map.json (discovered at dataset-prep time)
    _label_map_path = DATA_DIR / "label_map.json"
    if _label_map_path.exists():
        with open(_label_map_path, encoding="utf-8") as _f:
            _lm = json.load(_f)
        num_labels = _lm["num_labels"]  # 92 (O + 91 POS tags)
    else:
        num_labels = 91  # fallback

    # Training
    batch_size      = 192       # larger batches for better GPU utilization
    gradient_accum  = 6         # effective batch_size = 1152
    lr_head         = 5e-4
    lr_encoder      = 2e-5       # standard BERT fine-tuning LR
    weight_decay    = 0.01
    warmup_ratio    = 0.1        # warmup for 1M samples
    max_epochs      = 30
    early_stop_pat  = 3          # epochs without improvement
    unfreeze_layers = 6         # unfreeze last 6 encoder layers
    use_fp16        = True       # mixed precision for speed

    # Case-particle weighting — detect dynamically from tag names
    case_weight     = 2.0

    # Focal Loss — focuses on hard examples (misclassified / rare labels)
    focal_gamma     = 2.0   # 0 = standard CE

    # CRF Layer — models label transition constraints (e.g. case.gen → n.count ✓)
    use_crf         = True  # BERT-CRF instead of BERT-Linear

    # ENS class weights — prevents extreme rare-class weights (β=0.9999 caps at ~1000x)
    ens_beta        = 0.9999

    # Supervised Contrastive Loss — auxiliary task for better token embeddings
    use_contrastive   = True
    contrastive_weight = 0.1    # λ for combining with focal/CRF loss
    contrastive_temp   = 0.1    # temperature for SupCon

    # Data
    max_len           = 512
    max_train_samples = 300000  # 300K samples
    max_eval_samples  = 10000   # smaller eval set for faster epoch evaluation
    seed              = 42
    drop_punct_prob   = 0.3   # 30% of ་/། tokens → UNK (data augmentation)

    # Logging
    log_interval    = 100       # steps


# ── Dataset ────────────────────────────────────────────────────────────────────

# Punctuation token IDs from ClassicalTibetanTokenizer
_TSEG_ID = 226   # ་
_SHAD_ID = 5     # །


class PosDataset(Dataset):
    """
    Memory-mapped POS dataset with optional punctuation-drop augmentation
    and rare-tag oversampling.
    """

    def __init__(
        self,
        split: str,
        data_dir: Path = DATA_DIR,
        max_samples: int = None,
        drop_punct_prob: float = 0.0,
        oversample_rare_tags: bool = False,
    ):
        self.drop_punct_prob = drop_punct_prob
        self.input_ids_full = np.load(data_dir / f"{split}_input_ids.npy",  mmap_mode="r")
        self.labels_full    = np.load(data_dir / f"{split}_labels.npy",     mmap_mode="r")

        total = len(self.input_ids_full)
        if max_samples and max_samples < total:
            self.size = max_samples
            self.input_ids = self.input_ids_full[:max_samples]
            self.labels    = self.labels_full[:max_samples]
        else:
            self.size = total
            self.input_ids = self.input_ids_full
            self.labels    = self.labels_full

        self.max_len = min(self.input_ids.shape[1], Config.max_len)

        # Rare-tag oversampling: build index that repeats samples containing hard tags
        self.oversample_base_indices = None
        if oversample_rare_tags and split == "train":
            self._build_rare_indices()
        else:
            self._base_len = self.size

    def _build_rare_indices(self):
        """Build index array that oversamples sentences containing rare tags."""
        OVERRARE = {"adj", "n.prop"}
        MINRARE  = {"v.past.v.pres", "v.fut.v.pres", "v.fut.v.past", "skt"}
        OVER_FACTOR = 5
        MIN_FACTOR  = 3

        # Load label map to get tag name → id mapping
        with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
            lm = json.load(f)
        tag_to_id = {v: int(k) for k, v in lm["id_to_label"].items()}
        over_ids = {tag_to_id[t] for t in OVERRARE if t in tag_to_id}
        min_ids  = {tag_to_id[t] for t in MINRARE  if t in tag_to_id}

        n = self.size
        base_indices = []
        extra_indices = []

        print(f"  Scanning {n:,} sentences for rare tags...", end=" ", flush=True)
        step = max(1, n // 1000)
        for i in range(n):
            labs = self.labels[i]
            valid = labs != -100
            tags_in = set(labs[valid].tolist())
            if   tags_in & over_ids: extra_indices.extend([i] * (OVER_FACTOR - 1))
            elif tags_in & min_ids:  extra_indices.extend([i] * (MIN_FACTOR  - 1))
            if i > 0 and i % step == 0:
                print(".", end="", flush=True)
        print(" done", flush=True)

        self._base_len = n
        self._extra_indices = extra_indices
        self._total_len = n + len(extra_indices)
        n_over = sum(1 for i in extra_indices if i in {j for j in range(n) if (set(self.labels[j][self.labels[j]!=-100].tolist()) & over_ids)})
        print(f"  Oversampling: base={n:,} + extra={len(extra_indices):,} → total={self._total_len:,}")

    def __len__(self) -> int:
        if getattr(self, '_extra_indices', None) is not None:
            return self._total_len
        return self.size

    def __getitem__(self, idx: int) -> dict:
        extra = getattr(self, '_extra_indices', None)
        if extra is not None:
            base_len = getattr(self, '_base_len', self.size)
            if idx < base_len:
                real_idx = idx
            else:
                real_idx = extra[idx - base_len]
        else:
            real_idx = idx

        ids  = torch.tensor(self.input_ids[real_idx, :self.max_len].copy(), dtype=torch.long)
        labs = torch.tensor(self.labels[real_idx,    :self.max_len].copy(), dtype=torch.long)

        # Punctuation-drop augmentation (only during training)
        if self.drop_punct_prob > 0:
            ids, labs = _drop_punct_augment(ids, labs, self.drop_punct_prob)

        return {"input_ids": ids, "labels": labs}


def _drop_punct_augment(ids: torch.Tensor, labs: torch.Tensor, prob: float) -> tuple:
    """
    Randomly replace ་/། tokens with UNK (ID 1) and ignore their labels.

    The model sees both punctuated and unpunctuated forms during training,
    so it learns robust representations that transfer to raw Tibetan text.
    """
    mask = (ids == _TSEG_ID) | (ids == _SHAD_ID)          # which positions are punct
    drop = mask & (torch.rand(ids.size(0)) < prob)           # which to drop
    ids  = ids.masked_fill(drop, 1)                           # → UNK (ID 1)
    labs = labs.masked_fill(drop, -100)                       # → ignore in loss
    return ids, labs


# ── Model ──────────────────────────────────────────────────────────────────────

class PosClassifier(nn.Module):
    """
    BERT encoder + token classification head for POS tagging.
    Loads TiBERT-classical-spm-80k encoder weights.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load pre-trained BERT encoder
        vocab_size = getattr(config, 'vocab_size', 80007)
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=config.max_len,
            pad_token_id=0,
        )
        self.bert = BertModel(bert_config)

        # Load pre-trained weights
        ckpt_path = MODEL_DIR / "model.safetensors"
        if ckpt_path.exists():
            print(f"  Loading encoder weights from {ckpt_path}")
            state_dict = {}
            try:
                from safetensors.torch import load_file
                sd = load_file(str(ckpt_path))
                for k, v in sd.items():
                    if k.startswith("bert."):
                        state_dict[k] = v
            except Exception:
                # Fallback: load with torch (weights_only=False for safetensors compat)
                sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                for k, v in sd.items():
                    if k.startswith("bert."):
                        state_dict[k] = v
            missing, unexpected = self.bert.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys (MLM head): {len(missing)} — expected (MLM head not needed)")
            print(f"  Loaded BERT encoder")

        # Partial freeze: freeze all, then unfreeze last N transformer layers
        for p in self.bert.parameters():
            p.requires_grad = False

        n_layers = config.unfreeze_layers  # unfreeze last N layers
        if n_layers > 0:
            # Unfreeze embeddings
            for p in self.bert.embeddings.parameters():
                p.requires_grad = True
            # Unfreeze last N encoder layers
            for i in range(12 - n_layers, 12):
                for p in self.bert.encoder.layer[i].parameters():
                    p.requires_grad = True
            frozen_params = sum(1 for p in self.bert.parameters() if not p.requires_grad)
            trainable_params = sum(1 for p in self.bert.parameters() if p.requires_grad)
            print(f"  Encoder partially frozen: {frozen_params} frozen, {trainable_params} trainable "
                  f"(last {n_layers} layers + embeddings)")

        # Token classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # CRF layer (models label transition constraints)
        self.use_crf = config.use_crf
        if self.use_crf:
            self.crf = CRF(config.num_labels)
            print(f"  CRF layer enabled (transitions + Viterbi decoding)")

        # Supervised Contrastive head (768 → 128, L2 normalized)
        self.use_contrastive = config.use_contrastive
        if self.use_contrastive:
            self.contrastive_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 128),
            )
            self.supcon_loss = SupervisedContrastiveLoss(temperature=config.contrastive_temp)
            print(f"  Contrastive head: 768→128 (SupCon loss λ={config.contrastive_weight})")

        # ENS class weights: prevents extreme rare-class weights
        # Effective Number of Samples: E_n = (1 - β^n) / (1 - β)
        # weight = 1 / E_n, then normalize and scale
        ens_beta = config.ens_beta
        ens_weights = torch.ones(config.num_labels, dtype=torch.float32)
        label_map_path = DATA_DIR / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as f:
                lm = json.load(f)
            id_to_label = {int(k): v for k, v in lm["id_to_label"].items()}
            # Build class counts from label_map counter (written by prepare_pos_dataset.py)
            class_counts = [lm["label_to_id"].get(id_to_label[i], 0) for i in range(config.num_labels)]
            # ENS
            ens = [(1 - ens_beta ** max(c, 1)) / (1 - ens_beta) for c in class_counts]
            inv_ens = [1.0 / e for e in ens]
            total = sum(inv_ens)
            base = [w / total * config.num_labels for w in inv_ens]
            ens_weights = torch.tensor(base, dtype=torch.float32)
            # Case-particle boost (叠加在 ENS 之上)
            case_count = 0
            for lid, tag in id_to_label.items():
                if tag.startswith("case."):
                    ens_weights[lid] *= config.case_weight
                    case_count += 1
            print(f"  ENS weights: β={ens_beta} (max ~1000x boost)")
            print(f"  Case-particle boost: {case_count} tags ×{config.case_weight}")
        self.register_buffer("loss_weights", ens_weights)

        print(f"  Classifier head: Linear({config.hidden_size} → {config.num_labels})")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq, 768]
        dropped = self.dropout(sequence_output)
        logits = self.classifier(dropped)   # [batch, seq, num_labels]

        loss = None
        if labels is not None:
            if self.use_crf:
                crf_mask = (labels != -100) & (attention_mask == 1)
                crf_labels = labels.clone()
                crf_labels[labels == -100] = 0
                crf_mask = crf_mask.bool()
                crf_ll = self.crf(logits, crf_labels, mask=crf_mask)
                main_loss = (-crf_ll).mean()
            else:
                loss_fct = FocalLoss(
                    gamma=self.config.focal_gamma,
                    weight=self.loss_weights,
                    ignore_index=-100,
                )
                main_loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            # Supervised Contrastive loss on clean embeddings
            if self.use_contrastive:
                valid_mask = (labels != -100) & (attention_mask == 1)
                proj = self.contrastive_head(sequence_output)   # [B, L, 128] L2 norm
                con_loss = self.supcon_loss(
                    proj.view(-1, proj.size(-1)),   # [N, 128]
                    labels,                         # [B, L]
                    valid_mask,                     # [B, L]
                )
                loss = main_loss + self.config.contrastive_weight * con_loss
            else:
                loss = main_loss

        return {"loss": loss, "logits": logits}


# ── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for token-level POS classification.

    FL(p_t) = -(1 - p_t)^gamma * CE(p_t)
    - gamma > 0: down-weights well-classified examples, focuses on hard ones
    - weight tensor: class-frequency weights (e.g. case.particle ×2)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B*L, num_labels], targets: [B*L]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce)                              # p_t for correct class
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss on token embeddings (SupCon).

    For each anchor token i with label L_i, pulls together all tokens with
    the same label and pushes apart tokens with different labels.

    loss = -sum_i log ( sum_j exp(sim(i,j)/τ) ) / ( sum_k exp(sim(i,k)/τ) )
           where j in positives (same label), k in all (same label, incl. self)

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, proj, labels, valid_mask):
        """
        Args:
            proj:       [N, D] L2-normalized token embeddings
            labels:     [B, L] token labels (may contain -100)
            valid_mask: [B, L] bool, True where token is valid (not pad/-100)
        Returns:
            scalar loss
        """
        # Flatten to [N]
        valid_mask_flat = valid_mask.view(-1)
        labels_flat     = labels.view(-1)

        # Filter to valid positions
        proj   = proj[valid_mask_flat]          # [N_valid, D]
        labels = labels_flat[valid_mask_flat]    # [N_valid]

        if len(proj) < 2:
            return torch.tensor(0.0, device=proj.device, requires_grad=True)

        N = len(proj)

        # Pairwise cosine similarity (proj is L2-normalized → dot = cos)
        # [N, D] @ [D, N] = [N, N]
        sim = proj @ proj.T / self.tau          # [N, N]

        # Numerical stability: subtract row-max before exp
        sim = sim - sim.max(dim=1, keepdim=True)[0]  # [N, N]

        # Labels as matrix: same[i,j] = 1 iff labels[i] == labels[j] != -100
        labels_i = labels.unsqueeze(0).expand(N, N)
        labels_j = labels.unsqueeze(1).expand(N, N)
        same = (labels_i == labels_j) & (labels_i != -100)   # [N, N]

        # Exclude self from positive mask
        pos_mask = same & ~torch.eye(N, dtype=torch.bool, device=proj.device)

        exp_sim = sim.exp()   # [N, N], now numerically stable

        # Numerator: sum of exp(sim) for positive pairs
        pos_num = (exp_sim * pos_mask.float()).sum(dim=1)   # [N]

        # Denominator: all same-label pairs (incl. self)
        denom_sim = (exp_sim * same.float()).sum(dim=1)      # [N]
        denom_sim = denom_sim.clamp(min=1e-8)

        # If a class has only 1 sample → pos_num=0 → log(eps) ≈ -11.5 → finite
        loss = (-torch.log(pos_num / denom_sim + 1e-8)).mean()
        return loss


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_token_metrics(logits: torch.Tensor, labels: torch.Tensor, id_to_label: dict):
    """
    Compute token-level precision/recall/F1 per label.
    If logits has argmax-dim (num_labels), argmax is applied.
    If logits is already decoded (no num_labels dim), used directly.
    """
    mask = labels != -100
    # If logits has a num_labels dimension, argmax. Otherwise treat as preds.
    preds = logits.argmax(dim=-1) if logits.dim() == labels.dim() + 1 else logits

    valid_preds = preds[mask].cpu().numpy()
    valid_labels = labels[mask].cpu().numpy()

    # Per-label stats
    label_stats = {}
    for lid in np.unique(valid_labels):
        lbl = id_to_label.get(lid, str(lid))
        tp = ((valid_preds == lid) & (valid_labels == lid)).sum()
        fp = ((valid_preds == lid) & (valid_labels != lid)).sum()
        fn = ((valid_labels == lid) & (valid_preds != lid)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support   = int((valid_labels == lid).sum())
        label_stats[lid] = {
            "label": lbl,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = (valid_preds == valid_labels).mean()
    return accuracy, label_stats


# ── Training ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: PosClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Config,
    device: str,
    epoch: int,
    scaler: GradScaler = None,
):
    model.train()
    total_loss = 0.0
    step = 0
    optimizer.zero_grad()
    ema_loss = None   # exponential moving average for display
    t0 = time.time()
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{config.max_epochs}",
        unit="batch",
        ncols=80,
        total=len(loader),
    )
    for batch in pbar:
        ids  = batch["input_ids"].to(device)
        labs = batch["labels"].to(device)
        attention_mask = (ids != 0).long()

        if config.use_fp16 and scaler is not None:
            with autocast('cuda'):
                outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labs)
            loss_per_token = outputs["loss"].mean() / config.gradient_accum
            scaler.scale(loss_per_token).backward()
        else:
            outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labs)
            loss_per_token = outputs["loss"].mean() / config.gradient_accum
            loss_per_token.backward()

        # EMA for display (smoothed per-batch loss, not cumulative)
        step_loss = outputs["loss"].mean().item()
        if ema_loss is None:
            ema_loss = step_loss
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * step_loss

        total_loss += loss_per_token.item() * config.gradient_accum

        if (step + 1) % config.gradient_accum == 0:
            if config.use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        # Update progress bar with EMA loss
        pbar.set_postfix({
            "loss": f"{ema_loss:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.0e}",
        })

    # Flush remaining gradients
    if step % config.gradient_accum != 0:
        if config.use_fp16 and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    pbar.close()
    # total_loss accumulates outputs["loss"].mean() per optimizer step
    # step // gradient_accum = number of optimizer steps actually taken
    num_opt_steps = step // config.gradient_accum
    avg_loss = total_loss / max(num_opt_steps, 1)
    elapsed = time.time() - t0
    return avg_loss, elapsed


@torch.no_grad()
def evaluate(
    model: PosClassifier,
    loader: DataLoader,
    config: Config,
    device: str,
    id_to_label: dict,
):
    model.eval()
    _m = model.module if isinstance(model, torch.nn.DataParallel) else model
    use_crf = _m.use_crf
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", unit="batch", ncols=80, total=len(loader))
    for batch in pbar:
        ids  = batch["input_ids"].to(device)
        labs = batch["labels"].to(device)
        attention_mask = (ids != 0).long()

        outputs = model(input_ids=ids, attention_mask=attention_mask)
        if use_crf:
            # Viterbi decoding: CRF considers transitions
            crf_mask = attention_mask.bool()
            decoded = _m.crf.viterbi_decode(outputs["logits"], mask=crf_mask)  # list of [L]
            # Pad all sequences to same length (max in batch)
            max_len = labs.size(1)
            padded = torch.full((len(decoded), max_len), 0, dtype=torch.long, device=device)
            for i, (seq_preds, seq_len) in enumerate(zip(decoded, attention_mask.sum(dim=1).tolist())):
                padded[i, :seq_len] = torch.tensor(seq_preds[:seq_len], dtype=torch.long, device=device)
            preds = padded
        else:
            preds = outputs["logits"].argmax(dim=-1)   # [B, L]

        all_preds.append(preds.cpu())
        all_labels.append(labs.cpu())
    pbar.close()

    all_preds   = torch.cat(all_preds,   dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)

    accuracy, label_stats = compute_token_metrics(all_preds, all_labels, id_to_label)
    return accuracy, label_stats


def print_classification_report(label_stats: dict, id_to_label: dict):
    """Print a readable classification report focusing on key labels."""
    print(f"\n{'Label':<20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Support':>8s}")
    print("-" * 55)

    # Sort by F1 descending
    sorted_stats = sorted(label_stats.values(), key=lambda x: -x["f1"])
    for s in sorted_stats:
        if s["support"] == 0:
            continue
        print(f"{s['label']:<20s} {s['precision']:>6.3f} {s['recall']:>6.3f} "
              f"{s['f1']:>6.3f} {s['support']:>8d}")

    # Macro & weighted F1
    non_zero = [s for s in label_stats.values() if s["support"] > 0]
    if non_zero:
        macro_f1 = sum(s["f1"] for s in non_zero) / len(non_zero)
        support_total = sum(s["support"] for s in non_zero)
        weighted_f1 = sum(s["f1"] * s["support"] for s in non_zero) / support_total
        print(f"\n  Macro F1:  {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")

        # Case particle F1
        case_stats = [s for s in non_zero if "case" in s["label"]]
        if case_stats:
            case_f1 = sum(s["f1"] * s["support"] for s in case_stats) / sum(s["support"] for s in case_stats)
            print(f"  Case-Particle Weighted F1: {case_f1:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(resume_from: int = 0, full_data: bool = False, max_train_samples_arg: int = None,
         max_epochs_override: int = None, data_dir_override: str = None, output_dir_override: str = None,
         oversample_rare: bool = False):
    global DATA_DIR, OUTPUT_DIR
    if data_dir_override:
        DATA_DIR = Path(data_dir_override)
    if output_dir_override:
        OUTPUT_DIR = Path(output_dir_override)
    cfg = Config()
    if max_train_samples_arg is not None:
        cfg.max_train_samples = max_train_samples_arg
    if full_data:
        cfg.max_train_samples = None   # use all training data
    if max_epochs_override is not None:
        cfg.max_epochs = max_epochs_override
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  TiBERT POS Classifier Training")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  GPUs:         {num_gpus}")
    print(f"  Encoder:      {'Frozen' if cfg.encoder_freeze else 'Fine-tuned'}")
    print(f"  Batch size:   {cfg.batch_size} × {cfg.gradient_accum} = {cfg.batch_size * cfg.gradient_accum} (effective)")
    if num_gpus > 1:
        print(f"  Per-GPU batch: {cfg.batch_size // num_gpus}")
    print(f"  Head LR:      {cfg.lr_head}")
    print(f"  Max epochs:  {cfg.max_epochs}")
    print(f"  Case weight:  {cfg.case_weight}× (ENS weighted)")
    print(f"  Focal gamma:  {cfg.focal_gamma}")
    print(f"  CRF layer:   {'enabled' if cfg.use_crf else 'disabled'}")
    print(f"  ENS beta:    {cfg.ens_beta} (caps rare-class weight at ~1000×)")
    print(f"  SupCon loss: {'enabled λ='+str(cfg.contrastive_weight) if cfg.use_contrastive else 'disabled'}")
    print(f"  Drop punct:   {cfg.drop_punct_prob*100:.0f}% of ་/། → UNK (augmentation)")
    print(f"  Rare oversample: {'enabled (5× adj/n.prop, 3× rare v.*)' if oversample_rare else 'disabled'}")
    print(f"  Early stop:   {cfg.early_stop_pat} epochs")

    # Load tokenizer
    print(f"\n[1] Loading tokenizer...")
    tok_path = MODEL_DIR / "spm.model"
    tokenizer = ClassicalTibetanTokenizer(spm_model_file=str(tok_path))
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load label map
    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        lm = json.load(f)
    id_to_label = {int(k): v for k, v in lm["id_to_label"].items()}
    print(f"  Labels: {cfg.num_labels}")

    # Create datasets
    print(f"\n[2] Loading datasets...")
    train_ds = PosDataset("train", max_samples=cfg.max_train_samples,
                           drop_punct_prob=cfg.drop_punct_prob,
                           oversample_rare_tags=oversample_rare)
    dev_ds   = PosDataset("dev",   max_samples=cfg.max_eval_samples)
    test_ds  = PosDataset("test",  max_samples=cfg.max_eval_samples)
    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    num_workers = min(2, num_gpus)
    persistent = num_workers > 0

    # Class-balanced WeightedRandomSampler: rare-class tokens appear more often
    print("  Computing class frequencies for balanced sampling...", end=" ", flush=True)
    labels_mmap = train_ds.labels_full[:len(train_ds)]   # mmap slice
    valid_mask  = labels_mmap != -100
    flat_labels = labels_mmap[valid_mask]
    class_counts = np.bincount(flat_labels, minlength=cfg.num_labels)
    # Each sample (sentence) weight = mean of its token class weights
    # → compute per-sentence weights by averaging 1/count over its valid tokens
    sample_weights = np.zeros(len(train_ds), dtype=np.float64)
    for i in range(len(train_ds)):
        sent_valid = labels_mmap[i] != -100
        sent_labels = labels_mmap[i][sent_valid]
        if len(sent_labels) > 0:
            sample_weights[i] = np.mean(1.0 / (class_counts[sent_labels] + 1))
    # Normalize
    sample_weights /= sample_weights.sum() * len(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    print(f"done (n_classes={np.count_nonzero(class_counts)})")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent, prefetch_factor=2)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size * 2, num_workers=num_workers,
                              pin_memory=True, persistent_workers=persistent, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size * 2, num_workers=num_workers,
                              pin_memory=True, persistent_workers=persistent, prefetch_factor=2)

    # Build model
    print(f"\n[3] Building model...")
    model = PosClassifier(cfg)

    # Multi-GPU: DataParallel across all available GPUs
    _is_parallel = num_gpus > 1
    if _is_parallel:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"  Using DataParallel on {num_gpus} GPUs")
    model.to(device)

    # Optimizer & scheduler
    _m = model.module if _is_parallel else model
    optimizer_params = [
        {"params": _m.classifier.parameters(), "lr": cfg.lr_head},
    ]
    # Contrastive head: same LR as classifier head
    if cfg.use_contrastive:
        optimizer_params.append({"params": _m.contrastive_head.parameters(), "lr": cfg.lr_head})
    # Encoder: only trainable (unfrozen) params with lower LR
    encoder_params = [p for p in _m.bert.parameters() if p.requires_grad]
    if encoder_params:
        optimizer_params.append({"params": encoder_params, "lr": cfg.lr_encoder})

    optimizer = AdamW(optimizer_params, weight_decay=cfg.weight_decay)

    # Mixed precision scaler
    scaler = GradScaler('cuda') if cfg.use_fp16 else None
    if cfg.use_fp16:
        print(f"  Mixed precision (FP16) enabled")

    num_training_steps = len(train_loader) // cfg.gradient_accum * cfg.max_epochs
    num_warmup_steps    = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    print(f"  Steps: {num_training_steps} (warmup: {num_warmup_steps})")

    # Training loop
    print(f"\n[4] Training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0
    # Load existing history if resuming so we don't lose prior entries
    history_path = OUTPUT_DIR / "training_history.json"
    history = []
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        print(f"  Loaded {len(history)} prior history entries")

    # Resume from checkpoint
    if resume_from > 0:
        ckpt_path = OUTPUT_DIR / f"checkpoint-epoch{resume_from}.pt"
        if ckpt_path.exists():
            print(f"  Resuming from epoch {resume_from} checkpoint...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Model was saved with DataParallel (module.xxx keys); load into model.module
            _m = model.module if isinstance(model, torch.nn.DataParallel) else model
            state_dict = ckpt["model_state"]
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            _m.load_state_dict(state_dict)
            # Reinit optimizer/scheduler — checkpoint optimizer was also DataParallel-wrapped
            num_training_steps = len(train_loader) // cfg.gradient_accum * cfg.max_epochs
            num_warmup_steps    = int(num_training_steps * cfg.warmup_ratio)
            optimizer = AdamW(optimizer_params, weight_decay=cfg.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            best_f1 = ckpt.get("dev_weighted_f1", 0.0)
            print(f"  Loaded model (epoch {resume_from}, F1={best_f1:.4f}), reinit optimizer/scheduler")
        else:
            print(f"  WARNING: checkpoint {ckpt_path} not found, starting from scratch")

    for epoch in range(resume_from + 1, cfg.max_epochs + 1):
        avg_loss, epoch_time = train_epoch(model, train_loader, optimizer, scheduler, cfg, device, epoch, scaler)

        # Evaluate
        dev_acc, dev_stats = evaluate(model, dev_loader, cfg, device, id_to_label)
        dev_weighted_f1 = (
            sum(s["f1"] * s["support"] for s in dev_stats.values() if s["support"] > 0)
            / max(sum(s["support"] for s in dev_stats.values() if s["support"] > 0), 1)
        )

        # Evaluate on test set
        test_acc, test_stats = evaluate(model, test_loader, cfg, device, id_to_label)
        test_weighted_f1 = (
            sum(s["f1"] * s["support"] for s in test_stats.values() if s["support"] > 0)
            / max(sum(s["support"] for s in test_stats.values() if s["support"] > 0), 1)
        )
        test_macro_f1 = sum(s["f1"] for s in test_stats.values() if s["support"] > 0) / max(sum(1 for s in test_stats.values() if s["support"] > 0), 1)

        print(f"\n  Epoch {epoch}/{cfg.max_epochs} | Loss {avg_loss:.4f} | Dev Acc {dev_acc:.4f} | "
              f"Dev WF1 {dev_weighted_f1:.4f} | Test Acc {test_acc:.4f} | Test WF1 {test_weighted_f1:.4f} | {epoch_time:.0f}s")

        # Save per-epoch test results
        test_history_entry = {
            "epoch": epoch,
            "test_accuracy": test_acc,
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
            "dev_acc": dev_acc,
            "dev_weighted_f1": dev_weighted_f1,
            "label_stats": {str(k): v for k, v in test_stats.items()},
        }
        test_history_path = OUTPUT_DIR / "test_history.json"
        if test_history_path.exists():
            with open(test_history_path, encoding="utf-8") as f:
                test_history = json.load(f)
        else:
            test_history = []
        # Keep last 30 entries to avoid huge files while preserving history
        test_history.append(test_history_entry)
        test_history = test_history[-30:]
        with open(test_history_path, "w", encoding="utf-8") as f:
            json.dump(test_history, f, indent=2, ensure_ascii=False)

        # Case-particle F1
        case_stats = [s for s in dev_stats.values() if s["support"] > 0 and "case" in s["label"]]
        if case_stats:
            case_f1 = sum(s["f1"] * s["support"] for s in case_stats) / max(sum(s["support"] for s in case_stats), 1)
            print(f"  Case-Particle WF1: {case_f1:.4f}")

        # Save checkpoint
        ckpt_path = OUTPUT_DIR / f"checkpoint-epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "dev_weighted_f1": dev_weighted_f1,
            "dev_acc": dev_acc,
            "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        }, ckpt_path)
        print(f"  Saved: {ckpt_path.name}")

        # Early stopping based on weighted F1
        if dev_weighted_f1 > best_f1:
            best_f1 = dev_weighted_f1
            patience_counter = 0
            # Save best
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
            }, OUTPUT_DIR / "best_model.pt")
            print(f"  ★ New best model! F1={best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg.early_stop_pat})")
            if patience_counter >= cfg.early_stop_pat:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev_acc": dev_acc,
            "dev_weighted_f1": dev_weighted_f1,
            "epoch_time_s": epoch_time,
        })
        # Incrementally save so no data is lost on interruption
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    # Final evaluation on test set
    print(f"\n[5] Final evaluation on test set...")
    if (OUTPUT_DIR / "best_model.pt").exists():
        ckpt = torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_f1_loaded = ckpt.get("dev_weighted_f1", best_f1)
        print(f"  Loaded best model (epoch {ckpt['epoch']}, F1={best_f1_loaded:.4f})")

    test_acc, test_stats = evaluate(model, test_loader, cfg, device, id_to_label)
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print_classification_report(test_stats, id_to_label)

    # Save final test results
    results = {
        "test_accuracy": test_acc,
        "best_dev_weighted_f1": best_f1,
        "label_stats": {str(k): v for k, v in test_stats.items()},
    }
    with open(OUTPUT_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from this epoch number (e.g. 5 = continue from epoch 5)")
    parser.add_argument("--full-data", action="store_true", help="Use full training dataset (overrides max_train_samples)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Override max training samples")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset directory (default: data/corpus/pos_dataset)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory (default: model/pos_classifier)")
    parser.add_argument("--oversample-rare", action="store_true", help="Oversample adj/n.prop/v.rare sentences 5× in training DataLoader")
    args = parser.parse_args()
    main(resume_from=args.resume_from, full_data=args.full_data,
         max_train_samples_arg=args.max_train_samples,
         max_epochs_override=args.max_epochs,
         data_dir_override=args.data_dir,
         output_dir_override=args.output_dir,
         oversample_rare=args.oversample_rare)
