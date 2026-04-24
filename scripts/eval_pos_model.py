"""
Evaluate a POS classifier checkpoint on the full test set.
Usage:
  .venv/bin/python scripts/eval_pos_model.py [--checkpoint PATH] [--max-samples N]
"""
import argparse, json, sys, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_pos_classifier import PosClassifier

DATA_DIR  = Path(__file__).parent.parent / "data" / "corpus" / "pos_dataset"
CKPT_PATH = Path(__file__).parent.parent / "model" / "pos_classifier" / "checkpoint-epoch12.pt"


class Cfg:
    hidden_size = 768; encoder_freeze = False; num_labels = 77
    batch_size = 128; gradient_accum = 16; lr_head = 5e-4; lr_encoder = 2e-5
    weight_decay = 0.01; warmup_ratio = 0.1; max_epochs = 20; early_stop_pat = 3
    unfreeze_layers = 6; use_fp16 = True; case_weight = 2.0; focal_gamma = 2.0
    use_crf = True
    use_contrastive = True
    contrastive_temp = 0.1
    contrastive_weight = 0.1
    ens_beta = 0.9999
    supcon_criterion = "supcon"
    max_len = 512; seed = 42; drop_punct_prob = 0.3


class EvalDS(Dataset):
    def __init__(self, split, max_len=512):
        self.ids  = np.load(DATA_DIR / f"{split}_input_ids.npy",  mmap_mode="r")
        self.labs = np.load(DATA_DIR / f"{split}_labels.npy",      mmap_mode="r")
        self.max_len = max_len
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i, :self.max_len].copy(), dtype=torch.long),
            "labels":    torch.tensor(self.labs[i, :self.max_len].copy(), dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT_PATH))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    with open(DATA_DIR / "label_map.json") as f:
        lm = json.load(f)
    id_to_label = {int(k): v for k, v in lm["id_to_label"].items()}
    num_labels   = lm["num_labels"]

    cfg = Cfg()
    cfg.num_labels = num_labels   # override with actual count

    print(f"Loading model from {args.checkpoint} ...")
    model = PosClassifier(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    missing_nocrf = [m for m in result.missing_keys if "crf" not in m]
    print(f"  Epoch:      {ckpt.get('epoch', '?')}")
    dev_wf1 = ckpt.get('dev_weighted_f1', '?')
    print(f"  Dev WF1:    {dev_wf1:.4f}" if isinstance(dev_wf1, (int, float)) else f"  Dev WF1:    {dev_wf1}")
    print(f"  Missing:    {missing_nocrf}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_ds = EvalDS(args.split)
    test_len = len(test_ds)
    if args.max_samples:
        test_len = min(args.max_samples, test_len)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)
    print(f"\nEvaluating {test_len:,} samples ...")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            ids  = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)
            mask = (ids != 0).long()
            out  = model(input_ids=ids, attention_mask=mask)
            preds = out["logits"].argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labs.cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    valid_mask   = all_labels != -100
    valid_preds  = all_preds[valid_mask]
    valid_labels = all_labels[valid_mask]

    total   = valid_labels.numel()
    correct = (valid_preds == valid_labels).sum().item()
    acc     = correct / total

    # Per-label stats
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for p, t in zip(valid_preds.tolist(), valid_labels.tolist()):
        lt = id_to_label.get(int(t), f"?{t}")
        lp = id_to_label.get(int(p), f"?{p}")
        stats[lt]["support"] += 1
        if lp == lt:
            stats[lt]["tp"] += 1
        else:
            stats[lt]["fn"] += 1
            stats[lp]["fp"] += 1

    results = []
    for label, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        results.append({"label": label, "precision": p, "recall": r, "f1": f, "support": s["support"]})

    total_support = sum(r["support"] for r in results)
    wf1 = sum(r["f1"] * r["support"] for r in results) / max(total_support, 1)
    mf1 = sum(r["f1"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  Test Accuracy:    {acc:.4f}  ({correct:,}/{total:,})")
    print(f"  Test Weighted F1: {wf1:.4f}")
    print(f"  Test Macro F1:    {mf1:.4f}")

    # Case particles
    case = sorted([r for r in results if r["label"].startswith("case.")], key=lambda x: -x["f1"])
    print(f"\nCase particles (n={sum(r['support'] for r in case):,}):")
    for r in case:
        icon = "★" if r["f1"] >= 0.95 else "●" if r["f1"] >= 0.80 else "○" if r["f1"] >= 0.50 else "✗"
        print(f"  {icon} {r['label']:<18} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  n={r['support']:,}")
    case_wf1 = sum(r["f1"]*r["support"] for r in case) / max(sum(r["support"] for r in case), 1)
    print(f"  Case-Particle WF1: {case_wf1:.4f}")

    # Top / bottom
    print(f"\nTop 15 (sup>=100):")
    for r in sorted(results, key=lambda x: -x["f1"])[:15]:
        if r["support"] < 100: continue
        print(f"  {r['label']:<22} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  n={r['support']:,}")

    print(f"\nBottom 15 (sup>=50):")
    for r in sorted(results, key=lambda x: x["f1"])[:15]:
        if r["support"] < 50: continue
        print(f"  {r['label']:<22} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  n={r['support']:,}")

    # Save
    out = {"test_acc": acc, "test_weighted_f1": wf1, "test_macro_f1": mf1,
           "case_particle_wf1": case_wf1, "per_label": results}
    p = Path(args.checkpoint)
    out_path = p.parent / f"eval_{p.stem}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
