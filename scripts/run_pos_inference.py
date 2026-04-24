"""
End-to-End POS Tagger Inference Demo
====================================
Loads the trained TiBERT + POS classifier and runs it on Tibetan text.

Usage:
    .venv/bin/python scripts/run_pos_inference.py "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
    .venv/bin/python scripts/run_pos_inference.py "དགེ་འདུན་གྱི་ཆོས་སྐུལ་དེ་ཡིན"
"""

import sys
import json
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from continued_pretrain import ClassicalTibetanTokenizer
from transformers import BertConfig, BertModel

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR   = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
CHECKPOINT  = Path(__file__).parent.parent / "model" / "pos_classifier" / "crf_supcon" / "best_model.pt"
LABEL_MAP   = Path(__file__).parent.parent / "model" / "pos_classifier" / "crf_supcon" / "test_results.json"
DICT_PATH   = Path(__file__).parent.parent / "data"  / "corpus" / "pos_dataset" / "dict_word_pos.pkl"

# ── Case particle metadata ─────────────────────────────────────────────────────

CASE_PARTICLES = {
    "case.gen":  "属格 (的 / of)",
    "case.agn":  "作格 (由 / by/agent)",
    "case.all":  "为格 (对 / for/to)",
    "case.abl":  "离格 (从 / from)",
    "case.ela":  "从格 (从 / from)",
    "case.ass":  "共同格 (与 / with)",
    "case.term": "终结格 (至 / to)",
    "case.loc":  "处格 (在 / at/in)",
    "case.comp": "比格 (比 / than)",
    "case.nare": "连格 (则 / then)",
}

TAG_DESCRIPTIONS = {
    "O":         "Outside (非标签)",
    "n.count":   "名词 (普通名词)",
    "n.prop":    "专有名词",
    "n.rel":     "关系名词",
    "n.mass":    "物质名词",
    "v.past":    "过去时动词",
    "v.pres":    "现在时动词",
    "v.fut":     "将来时动词",
    "v.invar":   "不变词动词",
    "v.neg":     "否定动词",
    "v.aux":     "助动词",
    "v.imp":     "命令式动词",
    "v.cop":     "系词 (是/为)",
    "v.*.past":  "动词过去时词根",
    "v.*.pres":  "动词现在时词根",
    "v.*.fut":   "动词将来时词根",
    "n.v.*":     "名动复合词",
    "adj":       "形容词",
    "cv.*":      "连接动词",
    "cl.*":      "类标记",
    "d.*":       "限定词/冠词",
    "num.*":     "数词",
    "adv.*":     "副词",
    "punc":      "标点符号",
    "neg":       "否定词",
    "skt":       "梵语音译",
    "case.*":    "格助词",
}


# ── Dictionary Post-Processing ─────────────────────────────────────────────────

class DictPostProcessor:
    """
    Maximum Forward Matching (MMW) to reconstruct Tibetan compound words
    from the raw text, then override model predictions at the
    corresponding SPM token positions.

    The SPM tokenizer fragments multi-syllable compound words (e.g.
    བཅོམ་ལྡན་འདས → བཅོམ ་ ལྡན ་ འདས).  We bypass this by running MMW
    directly on the original Unicode text — which has intact word boundaries
    — then mapping those character ranges back to SPM token positions.
    """

    def __init__(self, dict_path: str | Path):
        with open(dict_path, 'rb') as f:
            self.word_pos = pickle.load(f)  # word → (tag, count)
        self.max_word_len = max(len(w) for w in self.word_pos) if self.word_pos else 0
        print(f"[DictPostProcessor] Loaded {len(self.word_pos)} words (max={self.max_word_len} chars)")

    def _build_spm_char_map(self, text: str, syllables: list[str]) -> list[tuple[int, int]]:
        """
        Map each SPM token to its character offset range in the raw text.
        Also compute byte offsets for UTF-8 encoding correctness.
        Returns list of (char_start, char_end) for each syllable token.
        """
        char_pos = 0
        char_ranges = []
        for tok in syllables:
            start = char_pos
            end = start + len(tok)   # len() counts Unicode chars, not bytes
            char_ranges.append((start, end))
            char_pos = end
        return char_ranges

    def _mmw_on_raw_text(self, text: str) -> list[tuple[int, int, str]]:
        """
        Two-pass MMW on raw Unicode text using character positions.
        Returns list of (char_start, char_end, word) for matched dictionary words.
        Separators (་/།) inside a matched word are allowed — Tibetan orthography
        includes them within word boundaries.
        """
        n_chars = len(text)

        # Pass 1: collect all valid dictionary matches at every character position
        candidates = []  # (char_length, char_start, char_end, word)
        for start in range(n_chars):
            ch = text[start]
            # Skip standalone separator markers
            if ch in ('་', '།'):
                continue
            max_search = min(self.max_word_len, n_chars - start)
            found = False
            for length in range(max_search, 0, -1):
                word = text[start:start + length]
                if word in self.word_pos:
                    candidates.append((length, start, start + length, word))
                    found = True
                    break
            if found:
                continue

        if not candidates:
            return []

        # Pass 2: longest-match-wins (by character count)
        candidates.sort(key=lambda x: -x[0])
        used_chars = set()
        result = []
        for char_length, start, end, word in candidates:
            chars_in_word = set(range(start, end)) & used_chars
            if chars_in_word:
                continue  # some chars already claimed by longer match
            result.append((start, end, word))
            used_chars.update(range(start, end))
        return result

    def override_predictions(
        self,
        raw_text: str,
        syllables: list[str],
        tagged_tokens: list,
        id_to_label: dict,
    ) -> list:
        """
        MMW on raw text → map word ranges to SPM token indices → override.
        """
        # Step 1: MMW on raw text
        word_matches = self._mmw_on_raw_text(raw_text)
        if not word_matches:
            return tagged_tokens

        # Step 2: build SPM token → character range map
        spm_char_map = self._build_spm_char_map(raw_text, syllables)

        # Step 3: for each word match, find which SPM tokens it contains
        # Separators (་/།) are never overridden — they're orthographic markers.
        spm_to_words = [[] for _ in syllables]
        for char_start, char_end, word in word_matches:
            tag, count = self.word_pos[word]
            if count < 2:
                continue
            for tok_idx, (spm_start, spm_end) in enumerate(spm_char_map):
                if tok_idx < len(syllables) and syllables[tok_idx] in ('་', '།'):
                    continue  # separators are never overridden
                if spm_start >= char_start and spm_end <= char_end:
                    spm_to_words[tok_idx].append((word, tag, count))

        # Step 4: apply overrides — prefer the longest character match
        # When multiple dictionary words cover the same SPM token (nested coverage),
        # the one with the most characters is most likely the correct word.
        overrides = {}  # spm_idx → dict_tag
        for tok_idx, word_list in enumerate(spm_to_words):
            if not word_list:
                continue
            best = max(word_list, key=lambda x: len(x[0]))  # longest by char count
            _, best_tag, best_count = best
            overrides[tok_idx] = best_tag

        if not overrides:
            return tagged_tokens

        changed = sum(
            1 for idx, new_tag in overrides.items()
            if idx < len(tagged_tokens) and tagged_tokens[idx].label != new_tag
        )
        print(f"  [Dict Override] {changed} syllables overridden:")
        for idx, new_tag in sorted(overrides.items()):
            if idx < len(tagged_tokens):
                old_tag = tagged_tokens[idx].label
                if old_tag != new_tag:
                    print(f"    [{tagged_tokens[idx].token}] {old_tag} → {new_tag}")

        # Apply overrides
        result = []
        for idx, tok in enumerate(tagged_tokens):
            if idx in overrides and tok.token not in ("་", "།"):
                new_label = overrides[idx]
                new_id = next((i for i, l in id_to_label.items() if l == new_label), tok.label_id)
                result.append(TaggedToken(
                    token=tok.token,
                    label_id=new_id,
                    label=new_label,
                    description=TAG_DESCRIPTIONS.get(new_label, new_label),
                    is_case_particle=new_label in CASE_PARTICLES,
                    case_meaning=CASE_PARTICLES.get(new_label) if new_label in CASE_PARTICLES else None,
                ))
            else:
                result.append(tok)
        return result


# ── Model ─────────────────────────────────────────────────────────────────────

class PosTagger(nn.Module):
    """TiBERT encoder + POS classification head."""

    def __init__(self, num_labels: int = 36, max_len: int = 512):
        super().__init__()
        self.bert = BertModel(BertConfig(
            vocab_size=32007,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=max_len,
            pad_token_id=0,
        ))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(out.last_hidden_state))
        return logits


# ── Inference ─────────────────────────────────────────────────────────────────

@dataclass
class TaggedToken:
    token: str
    label_id: int
    label: str
    description: str
    is_case_particle: bool
    case_meaning: Optional[str]


class PosInference:
    """End-to-end POS tagger."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = ClassicalTibetanTokenizer(
            spm_model_file=str(MODEL_DIR / "spm.model")
        )

        # Load label map from test_results.json (has correct label IDs from training)
        with open(LABEL_MAP, encoding="utf-8") as f:
            lm = json.load(f)
        # test_results.json stores {id: {"label": "n.count", "f1": ..., "support": ...}}
        label_stats = lm.get("label_stats", {})
        self.id_to_label = {}
        for k, v in label_stats.items():
            self.id_to_label[int(k)] = v["label"]
        # model was trained with 77 labels; fill in missing IDs with placeholders
        for i in range(77):
            if i not in self.id_to_label:
                self.id_to_label[i] = f"UNK_{i}"
        self.num_labels = 77  # checkpoint trained with 77 classes

        # Load model — checkpoint has 77 labels; use that number
        real_num_labels = 77
        self.model = PosTagger(num_labels=real_num_labels)
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
        state = ckpt["model_state"]
        # Strip module. (DataParallel) and bert. (module prefix) prefixes
        state = {k.replace("module.", "").replace("bert.", ""): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[PosInference] missing keys (expected for CRF/contrastive head): {len(missing)}")
        self.model.to(self.device)
        self.model.eval()
        print(f"[PosInference] Loaded BERT+Linear (skipping CRF/contrastive) epoch {ckpt['epoch']} → {self.device}")
        self.model.to(self.device)
        self.model.eval()
        print(f"[PosInference] Loaded model from epoch {ckpt['epoch']} → {self.device}")

        # Load dictionary post-processor
        if DICT_PATH.exists():
            self.dict_processor = DictPostProcessor(DICT_PATH)
        else:
            self.dict_processor = None
            print("[PosInference] No dict_word_pos.pkl found — skipping dict post-processing")

    def spm_tokenize(self, text: str) -> list[str]:
        """
        Split Tibetan text into syllable tokens, aligned with training pipeline.

        Training uses spm_tokenize_word() which strips shad markers from syllables
        (e.g. བོད་ → བོད + ་) before SentencePiece lookup.
        To avoid train-inference tokenization mismatch (34pp accuracy gap),
        we split by shad first, apply longest-match within each bare syllable,
        then re-attach the shad markers.
        """
        # Step 1: split by shad markers to preserve syllable boundaries
        shad_markers = []
        for ch in text:
            if ch in ('་', '།', '༔'):
                shad_markers.append(ch)
            else:
                shad_markers.append(None)

        # Step 2: longest-match within each syllable (not across them)
        result = []
        unk_id = self.tokenizer._token2id.get("[UNK]")
        i = 0
        while i < len(text):
            ch = text[i]
            # Shad markers are their own tokens
            if ch in ('་', '།', '༔'):
                result.append(ch)
                i += 1
                continue
            # Longest-match within a syllable
            best = None
            max_len = min(10, len(text) - i)
            for n in range(max_len, 0, -1):
                sub = text[i:i+n]
                # Stop if we hit a shad marker mid-span
                if any(c in ('་', '།', '༔') for c in sub):
                    continue
                tid = self.tokenizer._convert_token_to_id(sub)
                if tid != unk_id:
                    best = sub
                    break
            if best is None:
                best = text[i]
            result.append(best)
            i += len(best)
        return result

    def tag(self, text: str) -> list[TaggedToken]:
        """Tag a Tibetan text string."""
        syllable_tokens = self.spm_tokenize(text)

        # Encode
        ids = [self.tokenizer.bos_token_id]
        for tok in syllable_tokens:
            tid = self.tokenizer._convert_token_to_id(tok)
            ids.append(tid)
        ids.append(self.tokenizer.eos_token_id)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = (ids_tensor != 0).long()

        # Predict
        with torch.no_grad():
            logits = self.model(ids_tensor, attention_mask=attention_mask)
            pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        # Build result (skip BOS/EOS)
        # Note: ་/། are separator markers; we still tag them but flag specially
        results = []
        for tok, lid in zip(syllable_tokens, pred_ids[1:-1]):
            label = self.id_to_label.get(lid, "O")
            is_sep = tok in ("་", "།")  # syllable separator markers
            is_case = (not is_sep) and any(label.startswith(c) for c in CASE_PARTICLES)
            results.append(TaggedToken(
                token=tok,
                label_id=lid,
                label=label,
                description=TAG_DESCRIPTIONS.get(label, label),
                is_case_particle=is_case,
                case_meaning=CASE_PARTICLES.get(label) if is_case else None,
            ))

        # Dictionary post-processing: MMW word reconstruction + POS override
        if self.dict_processor is not None:
            results = self.dict_processor.override_predictions(
                text, syllable_tokens, results, self.id_to_label
            )

        return results


# ── Output Formatters ──────────────────────────────────────────────────────────

def print_tagged_line(tagged: list[TaggedToken]):
    """Print a formatted token-level annotation."""
    print()
    print("─" * 70)
    print("  藏文       │ 标签           │ 说明")
    print("─" * 70)
    for t in tagged:
        is_sep = t.token in ("་", "།")
        flag = " ★" if t.is_case_particle else (" ·" if is_sep else "")
        case = f" ({t.case_meaning})" if t.case_meaning else ""
        if is_sep:
            desc = "(音节分隔符)"
        else:
            desc = t.description if len(t.description) <= 18 else t.description[:16] + "…"
        print(f"  {t.token:<10s} │ {t.label:<14s} │ {desc}{case}{flag}")
    print("─" * 70)


def print_summary(tagged: list[TaggedToken]):
    """Print a summary of case particles found."""
    # Exclude separator markers from counts
    syllables = [t for t in tagged if t.token not in ("་", "།")]
    case_particles = [t for t in syllables if t.is_case_particle]
    nouns = [t for t in syllables if t.label.startswith("n.")]
    verbs = [t for t in syllables if t.label.startswith("v.")]
    adj = [t for t in syllables if t.label == "adj"]

    print(f"\n  ┌─ 词类统计")
    print(f"  │ 名词: {len(nouns)} | 动词: {len(verbs)} | 形容词: {len(adj)} | 格助词: {len(case_particles)}")
    print(f"  └─")

    if case_particles:
        print(f"\n  ┌─ 格助词详解")
        for t in case_particles:
            print(f"  │ [{t.token}] → {t.case_meaning}")
        print(f"  └─")

        # Build sentence structure
        structure = []
        for t in syllables:
            if t.is_case_particle:
                structure.append(f"[格助:{t.token}]")
            elif t.label.startswith("n."):
                structure.append(f"[名:{t.token}]")
            elif t.label.startswith("v."):
                structure.append(f"[动:{t.token}]")
            elif t.label == "adj":
                structure.append(f"[形:{t.token}]")
            else:
                structure.append(f"[{t.token}]")
        print(f"\n  句法结构: {' + '.join(structure)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TiBERT POS Tagger")
    parser.add_argument("text", nargs="?", default="བོད་གི་ཡུལ་ལྷོ་ལ་སོང་",
                        help="Classical Tibetan text to tag")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full details")
    args = parser.parse_args()

    tagger = PosInference()
    print(f"\n输入: {args.text}")
    print(f"分词: {' | '.join(tagger.spm_tokenize(args.text))}")

    tagged = tagger.tag(args.text)
    print_tagged_line(tagged)

    if args.verbose:
        print_summary(tagged)


if __name__ == "__main__":
    main()
