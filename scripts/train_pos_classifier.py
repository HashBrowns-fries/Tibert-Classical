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
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from torch.amp import GradScaler, autocast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
    batch_size      = 64        # reduced for larger dataset
    gradient_accum  = 4         # effective batch_size = 256
    lr_head         = 5e-4
    lr_encoder      = 2e-5       # standard BERT fine-tuning LR
    weight_decay    = 0.01
    warmup_ratio    = 0.1        # more warmup for larger dataset
    max_epochs      = 20
    early_stop_pat  = 8          # epochs without improvement
    unfreeze_layers = 6         # unfreeze last 6 encoder layers
    use_fp16        = True       # mixed precision for speed

    # Case-particle weighting — detect dynamically from tag names
    case_weight     = 2.0

    # Data
    max_len           = 512
    max_train_samples = None   # use full training set
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
    Memory-mapped POS dataset with optional punctuation-drop augmentation.

    ་/། augmentation: during training, each ་/། token is randomly replaced
    with UNK (ID 1) with probability `drop_punct_prob`. Its label is set to -100
    (ignored in loss). This teaches the model to handle texts without tsheg/shad
    markers — the kind of input that appears in unstructured Tibetan sources.
    """

    def __init__(
        self,
        split: str,
        data_dir: Path = DATA_DIR,
        max_samples: int = None,
        drop_punct_prob: float = 0.0,
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

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        ids  = torch.tensor(self.input_ids[idx, :self.max_len].copy(), dtype=torch.long)
        labs = torch.tensor(self.labels[idx,    :self.max_len].copy(), dtype=torch.long)

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
    Loads TiBERT-classical-spm-500k encoder weights.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load pre-trained BERT encoder
        bert_config = BertConfig(
            vocab_size=32007,
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

        # Loss weights: case particles ×2 (detect from tag names)
        self.register_buffer(
            "loss_weights",
            torch.ones(config.num_labels, dtype=torch.float32)
        )
        # Load label map to find case particle IDs
        label_map_path = DATA_DIR / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as f:
                lm = json.load(f)
            id_to_label = {int(k): v for k, v in lm["id_to_label"].items()}
            w = self.loss_weights.clone()
            for lid, tag in id_to_label.items():
                if tag.startswith("case."):
                    w[lid] = config.case_weight
            self.register_buffer("loss_weights", w)
            case_count = sum(1 for t in id_to_label.values() if t.startswith("case."))
            print(f"  Case-particle weighting: {case_count} tags ×{config.case_weight}")

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
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)   # [batch, seq, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.loss_weights,
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_token_metrics(logits: torch.Tensor, labels: torch.Tensor, id_to_label: dict):
    """
    Compute token-level precision/recall/F1 per label.
    Returns (accuracy, dict of per-label {label: {p, r, f1, support}})
    """
    mask = labels != -100
    preds = logits.argmax(dim=-1)

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
            loss = outputs["loss"] / config.gradient_accum
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labs)
            loss = outputs["loss"] / config.gradient_accum
            loss.backward()

        total_loss += loss.item() * config.gradient_accum

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

        # Update progress bar
        n_steps = max(step // config.gradient_accum, 1)
        pbar.set_postfix({
            "loss": f"{total_loss / n_steps:.4f}",
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
    avg_loss = total_loss / max(step // config.gradient_accum, 1)
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
    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", unit="batch", ncols=80, total=len(loader))
    for batch in pbar:
        ids  = batch["input_ids"].to(device)
        labs = batch["labels"].to(device)
        attention_mask = (ids != 0).long()

        outputs = model(input_ids=ids, attention_mask=attention_mask)
        all_logits.append(outputs["logits"].cpu())
        all_labels.append(labs.cpu())
    pbar.close()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy, label_stats = compute_token_metrics(all_logits, all_labels, id_to_label)
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

def main():
    cfg = Config()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  TiBERT POS Classifier Training")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Encoder:      {'Frozen' if cfg.encoder_freeze else 'Fine-tuned'}")
    print(f"  Batch size:   {cfg.batch_size} × {cfg.gradient_accum} = {cfg.batch_size * cfg.gradient_accum}")
    print(f"  Head LR:      {cfg.lr_head}")
    print(f"  Max epochs:  {cfg.max_epochs}")
    print(f"  Case weight:  {cfg.case_weight}×")
    print(f"  Drop punct:   {cfg.drop_punct_prob*100:.0f}% of ་/། → UNK (augmentation)")
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
    train_ds = PosDataset("train", max_samples=cfg.max_train_samples, drop_punct_prob=cfg.drop_punct_prob)
    dev_ds   = PosDataset("dev")
    test_ds  = PosDataset("test")
    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size * 2, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size * 2, num_workers=4)

    # Build model
    print(f"\n[3] Building model...")
    model = PosClassifier(cfg)
    model.to(device)

    # Optimizer & scheduler
    optimizer_params = [
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ]
    # Encoder: only trainable (unfrozen) params with lower LR
    encoder_params = [p for p in model.bert.parameters() if p.requires_grad]
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
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        avg_loss, epoch_time = train_epoch(model, train_loader, optimizer, scheduler, cfg, device, epoch, scaler)

        # Evaluate
        dev_acc, dev_stats = evaluate(model, dev_loader, cfg, device, id_to_label)
        dev_weighted_f1 = (
            sum(s["f1"] * s["support"] for s in dev_stats.values() if s["support"] > 0)
            / max(sum(s["support"] for s in dev_stats.values() if s["support"] > 0), 1)
        )

        print(f"\n  Epoch {epoch}/{cfg.max_epochs} | Loss {avg_loss:.4f} | Dev Acc {dev_acc:.4f} | "
              f"Dev WF1 {dev_weighted_f1:.4f} | {epoch_time:.0f}s")

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

    # Save history
    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
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
    main()
