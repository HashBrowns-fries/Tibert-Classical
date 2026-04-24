"""
Extract learning data from SegPOS corpus for the learner tool.

Usage:
  .venv/bin/python scripts/learner_corpus_analysis.py [--max-sents N] [--output JSON_PATH]

Outputs:
  - Case particle drill sentences (grouped by particle type)
  - Verb conjugation examples from corpus
  - High-frequency vocabulary (nouns, verbs)
  - Corpus statistics for learning
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SEGPOS_BASE = Path(__file__).parent.parent / "data" / "segpos_extracted"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "learner_data.json"

# ── Case particle patterns ───────────────────────────────────────────────────
CASE_PARTICLES = {
    "case.gen":  {"tibetan": "གི",  "name": "属格", "english": "genitive", "chinese": "的"},
    "case.agn":  {"tibetan": "གྱིས", "name": "作格", "english": "ergative-agentive", "chinese": "由"},
    "case.all":  {"tibetan": "ལ་",   "name": "为格", "english": "allative", "chinese": "对"},
    "case.abl":  {"tibetan": "ལས་", "name": "离格", "english": "ablative", "chinese": "从"},
    "case.ela":  {"tibetan": "ནས་", "name": "从格", "english": "elative", "chinese": "从…起"},
    "case.ass":  {"tibetan": "དང་", "name": "共同格", "english": "associative", "chinese": "与"},
    "case.term": {"tibetan": "དུ",  "name": "终结格", "english": "terminative", "chinese": "至"},
    "case.loc":  {"tibetan": "ན་",   "name": "处格", "english": "locative", "chinese": "在"},
}

# ── Parse SegPOS pos file ────────────────────────────────────────────────────
def parse_pos_sentence(line):
    """Parse SegPOS line → list of (word, tag)."""
    line = re.sub(r'\s*<utt>\s*$', '', line.strip())
    if not line:
        return []
    result = []
    for chunk in line.split():
        chunk = chunk.strip()
        if not chunk or chunk in ('<utt>', '<utt'):
            continue
        if '/' in chunk:
            parts = chunk.rsplit('/', 1)
            word = parts[0]
            tag = parts[1] if len(parts) == 2 else "xxx"
        else:
            word = chunk
            tag = "xxx"
        if not word or word in ('p1', 'p2', 'p3', 'p4', 'p5'):
            continue
        result.append((word, tag))
    return result


def is_case_particle(word, tag):
    """Return True if this is a standalone case particle."""
    return tag.startswith("case.")


def get_full_sentence_from_parsed(parsed):
    """Reconstruct the full Tibetan sentence string from parsed tokens."""
    parts = []
    for word, tag in parsed:
        parts.append(word)
    return "".join(parts)


def get_noun_head(particle_idx, parsed):
    """
    Find the noun that the case particle attaches to.
    particle_idx: index of the (word, tag) pair in parsed.
    Returns the noun word immediately before the particle.
    """
    # Case particle is usually the word AFTER the noun it attaches to
    # Look backwards from particle_idx
    for i in range(particle_idx - 1, -1, -1):
        word, tag = parsed[i]
        if tag in ("punc", "punc."):
            continue
        if tag.startswith("case.") or tag.startswith("cv.") or tag.startswith("cl."):
            continue
        if tag.startswith("d.") or tag.startswith("n."):
            return word
    return None


def get_particle_context(particle_idx, parsed, window=3):
    """Get surrounding words for context."""
    start = max(0, particle_idx - window)
    end = min(len(parsed), particle_idx + window + 1)
    return parsed[start:end], particle_idx - start  # (surrounding, position of particle in window)


# ── Load all pos files ───────────────────────────────────────────────────────
def load_all_sentences(max_sents=None):
    """Load all sentences from SegPOS pos/ directory."""
    all_files = list(SEGPOS_BASE.glob("*/**/pos/*.txt"))
    sentences = []
    collections = {}

    for fpath in all_files:
        if "__MACOSX" in str(fpath):
            continue
        coll = fpath.parent.parent.parent.name
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = parse_pos_sentence(line)
                if len(parsed) >= 3:
                    full = get_full_sentence_from_parsed(parsed)
                    sentences.append({
                        "parsed": parsed,
                        "full": full,
                        "collection": coll,
                    })
                    if coll not in collections:
                        collections[coll] = 0
                    collections[coll] += 1
                    if max_sents and len(sentences) >= max_sents:
                        return sentences, collections
    return sentences, collections


# ── Extract case particle examples ───────────────────────────────────────────
def extract_case_particles(sentences):
    """Extract sentences grouped by case particle type."""
    by_particle = defaultdict(list)
    all_verbs = defaultdict(list)  # verb stems → example sentences
    noun_freq = defaultdict(int)

    for sent in sentences:
        parsed = sent["parsed"]
        full = sent["full"]

        # Track noun frequency
        for word, tag in parsed:
            if tag.startswith("n."):
                noun_freq[word] += 1

        # Find case particles
        for i, (word, tag) in enumerate(parsed):
            if not is_case_particle(word, tag):
                continue

            context, pos_in_context = get_particle_context(i, parsed, window=4)
            context_words = "".join(w for w, _ in context)

            # Get the noun before this particle
            noun = get_noun_head(i, parsed)

            by_particle[tag].append({
                "particle": word.rstrip("/"),  # clean trailing /
                "particle_tag": tag,
                "noun": noun.rstrip("/") if noun else None,  # clean trailing /
                "sentence": full,
                "context": context_words,
                "pos_in_context": pos_in_context,
                "collection": sent["collection"],
            })

    return by_particle, dict(noun_freq)


# ── Extract verb examples ─────────────────────────────────────────────────────
def load_verbs_lexicon():
    """Load verb lemmas from lexicon."""
    verbs_file = Path(__file__).parent.parent / "dict" / "lexicon-of-tibetan-verb-stems-master" / "verbs.txt"
    verbs = {}

    if verbs_file.exists():
        with open(verbs_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if not parts:
                    continue
                verb = parts[0].strip()
                meaning = parts[1] if len(parts) > 1 else ""
                conjugation = parts[2] if len(parts) > 2 else ""
                verbs[verb] = {
                    "meaning": meaning,
                    "conjugation": conjugation,
                }
    return verbs


def extract_verb_examples(sentences, verbs_dict):
    """Extract verb forms from corpus sentences."""
    verb_forms = defaultdict(list)  # by tag → list of (word, full_sentence)
    lemmas_file = Path(__file__).parent.parent / "dict" / "lexicon-of-tibetan-verb-stems-master" / "lemmas.txt"

    # Build a fast lookup: word → set of possible tags
    word_to_tags = defaultdict(set)
    if lemmas_file.exists():
        with open(lemmas_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    word, tag = parts
                    word_to_tags[word.strip()].add(tag.strip())

    for sent in sentences:
        parsed = sent["parsed"]
        for word, tag in parsed:
            if not tag.startswith("v.") and not tag.startswith("n.v."):
                continue
            if word in ("་", "།"):
                continue
            if len(word) < 2:
                continue
            # Only add a limited number per tag
            if len(verb_forms[tag]) < 200:
                verb_forms[tag].append({
                    "form": word.rstrip("/"),  # clean trailing /
                    "sentence": sent["full"],
                    "lexicon_entry": verbs_dict.get(word, {}).get("meaning", ""),
                })

    return dict(verb_forms), dict(word_to_tags)


# ── Build frequency lists ─────────────────────────────────────────────────────
def build_frequency_lists(sentences):
    """Build frequency lists for learning."""
    noun_freq = defaultdict(int)
    verb_freq = defaultdict(int)
    adj_freq = defaultdict(int)
    word_total = 0

    for sent in sentences:
        for word, tag in sent["parsed"]:
            if word in ("་", "།", "༔"):
                continue
            word_total += 1
            if tag.startswith("n."):
                noun_freq[word] += 1
            elif tag.startswith("v."):
                verb_freq[word] += 1
            elif tag == "adj":
                adj_freq[word] += 1

    # Top 200 for each
    return {
        "nouns": sorted(noun_freq.items(), key=lambda x: -x[1])[:200],
        "verbs": sorted(verb_freq.items(), key=lambda x: -x[1])[:200],
        "adjectives": sorted(adj_freq.items(), key=lambda x: -x[1])[:100],
        "total_words": word_total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sents", type=int, default=None, help="Max sentences to process")
    ap.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = ap.parse_args()

    print("Loading SegPOS sentences...")
    sentences, collections = load_all_sentences(max_sents=args.max_sents)
    print(f"  Loaded {len(sentences):,} sentences from {len(collections)} collections")

    print("Loading verb lexicon...")
    verbs_dict = load_verbs_lexicon()
    print(f"  {len(verbs_dict)} verb lemmas loaded")

    print("Extracting case particle examples...")
    by_particle, noun_freq = extract_case_particles(sentences)
    for tag, items in sorted(by_particle.items()):
        info = CASE_PARTICLES.get(tag, {})
        print(f"  {tag} ({info.get('tibetan','?')}) — {len(items):,} examples")

    print("Extracting verb examples...")
    verb_forms, word_tags = extract_verb_examples(sentences, verbs_dict)
    print(f"  {sum(len(v) for v in verb_forms.values()):,} verb form examples")

    print("Building frequency lists...")
    freq_lists = build_frequency_lists(sentences)
    print(f"  {freq_lists['total_words']:,} total words, {len(freq_lists['nouns'])} nouns, {len(freq_lists['verbs'])} verbs")

    # ── Build output ───────────────────────────────────────────────────────────
    output = {
        "meta": {
            "total_sentences": len(sentences),
            "total_words": freq_lists["total_words"],
            "collections": collections,
        },
        "case_particles": {},
        "verbs": {},
        "frequency": freq_lists,
    }

    # Case particles with examples
    for tag, examples in by_particle.items():
        info = CASE_PARTICLES.get(tag, {})
        output["case_particles"][tag] = {
            **info,
            "count": len(examples),
            # Sample up to 50 for each particle (balanced)
            "examples": examples[:50],
        }

    # Verb forms
    for tag, forms in verb_forms.items():
        output["verbs"][tag] = {
            "count": len(forms),
            "examples": forms[:30],
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  Case particle types: {len(output['case_particles'])}")
    print(f"  Verb form types: {len(output['verbs'])}")


if __name__ == "__main__":
    main()
