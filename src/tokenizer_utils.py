"""Tokenizer helpers for compact IUPAC glycan strings."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import RobertaProcessing
from tokenizers.trainers import BpeTrainer
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast


# These patterns define the manual glycan parser. They are written separately so
# the parsing order stays explicit and can be adjusted without changing the main
# tokenization loop.
INLINE_LINKAGE_PATTERN = re.compile(r"[ab?][0-9?]-[0-9?]")
PAREN_LINKAGE_PATTERN = re.compile(r"\([ab?][0-9?]-[0-9?]\)")
PIPE_BRANCH_PATTERN = re.compile(r"\|[0-9?]+")
REDUCING_END_PATTERN = re.compile(r"\+aldi\b")

RESIDUE_PATTERN = re.compile(
    r"[A-Z][A-Za-z0-9,]{1,20}?"
    r"(?=(?:[ab?][0-9?]-[0-9?])|(?:[ab?](?=[()+|\[\]]|$))|(?:[()+|\[\]]|$)|(?:\+aldi))"
)

ROOT_ANOMER_PATTERN = re.compile(r"(?<=[A-Za-z0-9,])[ab?](?=[()+|\[\]]|$)")
SIMPLE_MODIFICATION_PATTERN = re.compile(r"(?:\d+[A-Za-z]{1,6}|[A-Za-z]{1,6})")


# The Hugging Face fast tokenizer backend cannot use Python lookaheads, so this
# explicit pattern mirrors the main biological units used by the manual parser.
HUGGINGFACE_FAST_PATTERN = (
    r"Neu5Gc9Ac|Neu5,9Ac2|ManNAc|GalNAc|GlcNAc|HexNAc|NeuAc|NeuGc|GlcA|IdoA|"
    r"GlcN|Fru|Fuc|Gal|Glc|Hex|Kdn|Man|Xyl|\d+S|[ab?][0-9?]-[0-9?]|\|[0-9?]+|[\(\)\[\]]|[ab]"
)

# The original GlyBERTa demo strings use parenthesized linkages such as
# ``Gal(b1-4)GlcNAc``. This project's corpus instead stores compact strings such
# as ``Galb1-4GlcNAc`` with inline linkage text. To preserve the same
# glyco-letter idea on this corpus, the adapted pattern isolates inline
# linkages plus branch delimiters, while leaving residue text between them
# untouched.
GLYBERTA_COMPACT_GLYCOLETTER_PATTERN = r"[ab?][0-9?]-[0-9?]|[\(\)\[\]]"

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@dataclass(frozen=True)
class GlycanToken:
    """Container for one parsed glycan token."""

    kind: str
    value: str


def _simple_parenthetical_content(glycan_string: str, start_index: int):
    """Return the contents of a simple parenthetical group.

    Nested parentheses return ``None`` because they need to be handled by the
    main parser as branch structure rather than as a simple inline modifier.
    """
    if glycan_string[start_index] != "(":
        return None

    depth = 0
    content_start = start_index + 1

    for index in range(start_index, len(glycan_string)):
        character = glycan_string[index]

        if character == "(":
            depth += 1
            if depth > 1:
                return None
        elif character == ")":
            depth -= 1
            if depth == 0:
                return glycan_string[content_start:index], index

    return None


def tokenize_compact_iupac(glycan_string: str):
    """Parse a compact IUPAC glycan string into typed tokens."""
    tokens = []
    index = 0

    while index < len(glycan_string):
        character = glycan_string[index]

        if character.isspace():
            index += 1
            continue

        match = PAREN_LINKAGE_PATTERN.match(glycan_string, index)
        if match:
            tokens.append(GlycanToken("linkage", match.group(0)))
            index = match.end()
            continue

        if character in "[]":
            kind = "branch_open" if character == "[" else "branch_close"
            tokens.append(GlycanToken(kind, character))
            index += 1
            continue

        if character == "(":
            simple_group = _simple_parenthetical_content(glycan_string, index)
            if simple_group is not None:
                content, end_index = simple_group
                if SIMPLE_MODIFICATION_PATTERN.fullmatch(content):
                    tokens.append(GlycanToken("branch_open", "("))
                    tokens.append(GlycanToken("modification", content))
                    tokens.append(GlycanToken("branch_close", ")"))
                    index = end_index + 1
                    continue

            tokens.append(GlycanToken("branch_open", "("))
            index += 1
            continue

        if character == ")":
            tokens.append(GlycanToken("branch_close", ")"))
            index += 1
            continue

        match = REDUCING_END_PATTERN.match(glycan_string, index)
        if match:
            tokens.append(GlycanToken("reducing_end", match.group(0)))
            index = match.end()
            continue

        match = PIPE_BRANCH_PATTERN.match(glycan_string, index)
        if match:
            tokens.append(GlycanToken("branch_position", match.group(0)))
            index = match.end()
            continue

        match = INLINE_LINKAGE_PATTERN.match(glycan_string, index)
        if match:
            tokens.append(GlycanToken("linkage", match.group(0)))
            index = match.end()
            continue

        match = ROOT_ANOMER_PATTERN.match(glycan_string, index)
        if match:
            tokens.append(GlycanToken("root_anomer", match.group(0)))
            index = match.end()
            continue

        match = RESIDUE_PATTERN.match(glycan_string, index)
        if match:
            tokens.append(GlycanToken("residue", match.group(0)))
            index = match.end()
            continue

        # Unmatched characters are kept so parsing failures remain visible
        # during inspection rather than being dropped silently.
        tokens.append(GlycanToken("unknown", character))
        index += 1

    return tokens


def split_glycan_string(glycan_string: str):
    """Return only token text values from the manual parser."""
    return [token.value for token in tokenize_compact_iupac(glycan_string)]


def split_glyberta_compact_string(glycan_string: str):
    """Split one compact glycan string with the GlyBERTa-style regex rule."""
    parts = re.split(f"({GLYBERTA_COMPACT_GLYCOLETTER_PATTERN})", glycan_string)
    return [part for part in parts if part and not part.isspace()]


def _coerce_vocab_tokens(vocab_or_tokenizer) -> set[str]:
    """Accept a vocab dict, token list, or HF tokenizer and return token text."""
    if hasattr(vocab_or_tokenizer, "get_vocab"):
        return set(vocab_or_tokenizer.get_vocab().keys())

    if isinstance(vocab_or_tokenizer, dict):
        return set(vocab_or_tokenizer.keys())

    return set(vocab_or_tokenizer)


def audit_oov_tokens(
    sequences,
    vocab_or_tokenizer,
    splitter,
    max_example_rows: int = 20,
    top_n_tokens: int = 25,
):
    """Return summary tables for tokens missing from one tokenizer vocabulary."""
    vocab_tokens = _coerce_vocab_tokens(vocab_or_tokenizer)

    total_tokens = 0
    total_oov_tokens = 0
    sequences_with_oov = 0
    oov_counter = Counter()
    example_rows = []

    for sequence in sequences:
        tokens = splitter(sequence)
        total_tokens += len(tokens)

        oov_tokens = [token for token in tokens if token not in vocab_tokens]
        if not oov_tokens:
            continue

        sequences_with_oov += 1
        total_oov_tokens += len(oov_tokens)
        oov_counter.update(oov_tokens)

        if len(example_rows) < max_example_rows:
            example_rows.append(
                {
                    "sequence": sequence,
                    "tokens": " | ".join(tokens[:40]),
                    "oov_tokens": " | ".join(oov_tokens),
                    "oov_count": len(oov_tokens),
                }
            )

    summary = {
        "num_sequences": len(sequences),
        "sequences_with_oov": sequences_with_oov,
        "sequence_oov_rate": sequences_with_oov / len(sequences) if sequences else 0.0,
        "total_tokens": total_tokens,
        "total_oov_tokens": total_oov_tokens,
        "token_oov_rate": total_oov_tokens / total_tokens if total_tokens else 0.0,
        "unique_oov_tokens": len(oov_counter),
    }

    oov_df = pd.DataFrame(
        [
            {"token": token, "count": count}
            for token, count in oov_counter.most_common(top_n_tokens)
        ]
    )
    examples_df = pd.DataFrame(example_rows)
    summary_df = pd.DataFrame([summary])

    return summary_df, oov_df, examples_df


def inspect_tokenizer(tokenizer, sample_glycan: str, tokenizer_name: str = "Tokenizer") -> None:
    """Print token text and integer IDs for one sample glycan."""
    print(f"\n--- Inspecting: {tokenizer_name} ---")
    print(f"Original String: {sample_glycan}")
    print("-" * 50)

    encoding = tokenizer.encode(sample_glycan)
    text_tokens = encoding.tokens
    id_tokens = encoding.ids

    print(f"{'Text Token':<15} | {'Integer ID'}")
    print("-" * 50)
    for text, token_id in zip(text_tokens, id_tokens):
        print(f"{text:<15} | {token_id}")

    print("-" * 50)
    print(f"Total tokens generated: {len(text_tokens)}\n")


def evaluate_tokenizer_stats(tokenizer_name: str, tokenized_sequences) -> None:
    """Print sequence-length and token-frequency summaries for one tokenizer."""
    seq_lengths = [len(sequence) for sequence in tokenized_sequences]
    avg_len = np.mean(seq_lengths)
    max_len = np.max(seq_lengths)

    all_tokens = [token for sequence in tokenized_sequences for token in sequence]
    unique_tokens_used = len(set(all_tokens))

    token_counts = Counter(all_tokens)
    top_10 = token_counts.most_common(10)

    print("\n==================================================")
    print(f"      STATISTICS: {tokenizer_name}")
    print("==================================================")
    print(f"Avg Tokens per Sequence:  {avg_len:.2f}")
    print(f"Max Tokens in a Sequence: {max_len}")
    print(f"Total Unique Tokens Used: {unique_tokens_used}")
    print("Top 10 Most Frequent Tokens:")
    print("-" * 50)
    for token, count in top_10:
        print(f"  {token:<15} : {count} occurrences")
    print()


def train_hybrid_char_bpe(train_file: str, vocab_size: int = 300, min_frequency: int = 2):
    """Train a BPE tokenizer directly on raw glycan strings.

    This variant intentionally leaves the pre-tokenizer unset so the merge rules
    are learned from the continuous glycan string rather than from a whitespace-
    or punctuation-split intermediate representation.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    tokenizer.train(files=[train_file], trainer=trainer)
    return tokenizer


def train_glyberta_wordlevel(train_file: str, max_length: int | None = None):
    """Train the project-adapted GlyBERTa-style WordLevel tokenizer.

    The tokenizer learns its vocabulary from the training split only. Instead
    of porting the original GlyBERTa split rule literally, this helper adapts
    the same "glyco-letter" idea to the compact IUPAC strings used in this
    project by isolating inline linkage text like ``b1-4`` and branch
    delimiters like ``(``, ``)``, ``[``, and ``]``.
    """
    with open(train_file, "r", encoding="utf-8") as file:
        train_sequences = [line.strip() for line in file if line.strip()]

    if not train_sequences:
        raise ValueError(f"No training sequences found in {train_file}")

    backend_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    backend_tokenizer.pre_tokenizer = Split(
        pattern=Regex(GLYBERTA_COMPACT_GLYCOLETTER_PATTERN),
        behavior="isolated",
    )

    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS)
    backend_tokenizer.train_from_iterator(train_sequences, trainer=trainer)
    backend_tokenizer.post_processor = RobertaProcessing(
        sep=("</s>", backend_tokenizer.token_to_id("</s>")),
        cls=("<s>", backend_tokenizer.token_to_id("<s>")),
    )

    if max_length is not None:
        backend_tokenizer.enable_truncation(max_length=max_length)

    tokenizer_kwargs = {
        "tokenizer_object": backend_tokenizer,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "sep_token": "</s>",
        "cls_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
    if max_length is not None:
        tokenizer_kwargs["model_max_length"] = max_length

    hf_tokenizer = PreTrainedTokenizerFast(**tokenizer_kwargs)

    return backend_tokenizer, hf_tokenizer
