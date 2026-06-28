"""Tokenizer helpers for compact IUPAC glycan strings."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


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
