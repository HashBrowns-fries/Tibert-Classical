"""
TBRC Tibetan Corpus Extractor
Extracts pure Tibetan text from TEI XML files and builds a structured corpus.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from typing import List, Dict, Set
import json

# TEI namespace
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Tibetan tsheg (word separator) and other punctuation to optionally remove
TIBETAN_TSHEG = "\u0f0b"      # ་ Tibetan tsheg (word separator)
TIBETAN_SHAD = "\u0f0d"       # ། Tibetan rnam bcad (sentence terminator)
TIBETAN_CIRCLE = "\u0f0c"     # ༌ Tibetan shad
TIBETAN_COMMA = "\u0f14"      # ༔

# Patterns for normalization
TSHEG_PATTERN = re.compile(rf"[{TIBETAN_TSHEG}]+")
SHAD_PATTERN = re.compile(rf"[{TIBETAN_SHAD}]+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


class TibetanCorpusExtractor:
    """Extracts and preprocesses Tibetan text from TEI XML files."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.collections = self._discover_collections()
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_characters": 0,
            "total_sentences": 0,
        }

    def _discover_collections(self) -> List[str]:
        """Discover all collection directories."""
        collections = []
        for item in self.root_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                collections.append(item.name)
        return sorted(collections)

    def extract_from_file(self, xml_path: Path, normalize: bool = True) -> List[str]:
        """Extract Tibetan text paragraphs from a single XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            paragraphs = []
            # Find all <p> elements in TEI namespace
            for p in root.iter(f"{{{TEI_NS['tei']}}}p"):
                text_parts = []
                # Get all text content including children
                for elem in p.iter():
                    if elem.text:
                        text_parts.append(elem.text)
                    if elem.tail:
                        text_parts.append(elem.tail)

                text = "".join(text_parts).strip()
                if text:
                    if normalize:
                        text = self.normalize_tibetan(text)
                    paragraphs.append(text)

            return paragraphs

        except ET.ParseError as e:
            print(f"  XML parse error in {xml_path}: {e}")
            return []
        except Exception as e:
            print(f"  Error processing {xml_path}: {e}")
            return []

    def normalize_tibetan(self, text: str) -> str:
        """
        Normalize Tibetan text.
        - Remove tsheg (word separator) if configured
        - Normalize whitespace
        - Keep Tibetan shad as sentence boundary
        """
        # Remove tsheg (word separator)
        text = TSHEG_PATTERN.sub("", text)
        # Normalize multiple spaces
        text = MULTI_SPACE_PATTERN.sub(" ", text)
        return text.strip()

    def extract_collection(self, collection_name: str, normalize: bool = True,
                          remove_tsheg: bool = True) -> Dict[str, List[str]]:
        """Extract all texts from a collection."""
        collection_dir = self.root_dir / collection_name
        if not collection_dir.exists():
            print(f"Collection not found: {collection_name}")
            return {}

        documents = {}
        xml_files = list(collection_dir.rglob("*.xml"))

        # Filter out __contents__.xml files
        xml_files = [f for f in xml_files if "__contents__" not in str(f)]

        print(f"\n[{collection_name}]")
        print(f"  Found {len(xml_files)} XML files")

        for xml_file in xml_files:
            # Get document ID from filename
            doc_id = xml_file.stem

            paragraphs = self.extract_from_file(xml_file, normalize=False)
            if paragraphs:
                # Apply normalization
                if remove_tsheg:
                    paragraphs = [self.normalize_tibetan(p) for p in paragraphs]
                documents[doc_id] = paragraphs
                self.stats["files_processed"] += 1
                self.stats["total_characters"] += sum(len(p) for p in paragraphs)
                self.stats["total_sentences"] += len(paragraphs)
            else:
                self.stats["files_failed"] += 1

        print(f"  Extracted {len(documents)} documents")
        return documents

    def extract_all(self, output_dir: str, normalize: bool = True,
                   remove_tsheg: bool = True) -> Dict[str, Dict]:
        """Extract texts from all collections and save to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_corpus = {}

        for collection in self.collections:
            documents = self.extract_collection(
                collection, normalize=normalize, remove_tsheg=remove_tsheg
            )
            if documents:
                all_corpus[collection] = documents

                # Save collection to individual JSON file
                collection_file = output_path / f"{collection}.json"
                with open(collection_file, "w", encoding="utf-8") as f:
                    json.dump(documents, f, ensure_ascii=False, indent=2)
                print(f"  Saved to {collection_file}")

        # Save combined corpus
        combined_file = output_path / "combined.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_corpus, f, ensure_ascii=False, indent=2)
        print(f"\nSaved combined corpus to {combined_file}")

        # Save statistics
        stats_file = output_path / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        return all_corpus

    def get_vocabulary(self, corpus: Dict[str, Dict]) -> Set[str]:
        """Extract unique vocabulary from corpus."""
        vocab = set()
        for collection_docs in corpus.values():
            for paragraphs in collection_docs.values():
                for para in paragraphs:
                    # Split by space (after tsheg removal) or just collect characters
                    tokens = para.split()
                    vocab.update(tokens)
        return vocab

    def print_stats(self):
        """Print extraction statistics."""
        print("\n" + "=" * 50)
        print("CORPUS EXTRACTION STATISTICS")
        print("=" * 50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files failed:     {self.stats['files_failed']}")
        print(f"Total characters: {self.stats['total_characters']:,}")
        print(f"Total sentences:  {self.stats['total_sentences']:,}")
        print("=" * 50)


def main():
    """Main extraction function."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract Tibetan corpus from TEI XML")
    parser.add_argument("input_dir", help="Input directory containing XML files")
    parser.add_argument("output_dir", help="Output directory for corpus files")
    parser.add_argument("--keep-tsheg", action="store_true",
                        help="Keep Tibetan tsheg (word separator)")
    parser.add_argument("--stats", action="store_true",
                        help="Print vocabulary statistics")

    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Keep tsheg: {args.keep_tsheg}")

    extractor = TibetanCorpusExtractor(args.input_dir)

    print(f"\nDiscovered {len(extractor.collections)} collections:")
    for c in extractor.collections:
        print(f"  - {c}")

    # Extract corpus
    corpus = extractor.extract_all(
        args.output_dir,
        normalize=True,
        remove_tsheg=not args.keep_tsheg
    )

    # Print statistics
    extractor.print_stats()

    # Vocabulary stats
    if args.stats:
        vocab = extractor.get_vocabulary(corpus)
        print(f"\nVocabulary size: {len(vocab):,}")

        # Character-level analysis
        all_text = ""
        for docs in corpus.values():
            for paras in docs.values():
                all_text += " ".join(paras) + " "

        # Count unique Tibetan characters
        tibetan_chars = set(c for c in all_text if 0x0f40 <= ord(c) <= 0x0fbc)
        print(f"Unique Tibetan characters: {len(tibetan_chars)}")


if __name__ == "__main__":
    main()
