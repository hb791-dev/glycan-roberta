"""Rule-based N-glycan structural classification for compact IUPAC sequences.

This helper supports a separate notebook workflow that classifies glycans from
their compact IUPAC strings rather than relying only on the existing label
table. The current goal is intentionally conservative:

- detect likely N-glycans from the canonical chitobiose + trimannose core
- subclass N-glycans into high mannose, hybrid, complex, or
  paucimannose/truncated when the structure is clear
- keep explicit ``unresolved`` outcomes instead of forcing weak assignments
- compare the structural calls against the existing project labels

The implementation is designed to stay lightweight and notebook-friendly. It
does not depend on the training stack used elsewhere in the project.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.notebook_utils import require_existing_path, stringify_path_values, validate_output_paths, write_json


ACCESSION_COLUMN = "glycan_id"
SEQUENCE_COLUMN = "sequence"
SPLIT_COLUMN = "split"
LABEL_JSON_COLUMN = "labels_json"
LABEL_LIST_COLUMN = "labels"

STRUCTURAL_CLASS_COLUMN = "structural_n_glycan_class"
STRUCTURAL_BINARY_COLUMN = "structural_is_n_glycan"
STRUCTURAL_SUBCLASS_COLUMN = "structural_n_glycan_subclass"
STRUCTURAL_REASON_COLUMN = "structural_assignment_reason"
STRUCTURAL_CONFIDENCE_COLUMN = "structural_assignment_confidence"
STRUCTURAL_CORE_STATUS_COLUMN = "structural_core_status"
STRUCTURAL_ARM3_STATUS_COLUMN = "structural_arm_a1_3_status"
STRUCTURAL_ARM6_STATUS_COLUMN = "structural_arm_a1_6_status"
STRUCTURAL_TOTAL_MAN_COLUMN = "structural_total_mannose_count"
STRUCTURAL_PARSE_STATUS_COLUMN = "structural_parse_status"

CURRENT_MAIN_CLASS_COLUMN = "current_main_glycan_class"
CURRENT_N_O_CATEGORY_COLUMN = "current_n_o_category"
CURRENT_PRIMARY_SUBTYPE_COLUMN = "current_primary_subtype_label"
CURRENT_SUBCLASS_COLUMN = "current_n_glycan_subclass_label"
CURRENT_SUBCLASS_MATCH_COUNT_COLUMN = "current_n_glycan_subclass_match_count"
CURRENT_LABEL_SIGNATURE_COLUMN = "current_label_signature"

STRUCTURAL_CLASS_ORDER = (
    "not_n_glycan",
    "n_glycan_high_mannose",
    "n_glycan_hybrid",
    "n_glycan_complex",
    "n_glycan_paucimannose_or_truncated",
    "n_glycan_unresolved",
)
STRUCTURAL_SUBCLASS_ORDER = (
    "Not N-glycan",
    "High mannose",
    "Hybrid",
    "Complex",
    "Paucimannose/truncated",
    "Unresolved N-glycan",
)

N_GLYCAN_SUBCLASS_KEYWORDS = {
    "High mannose": ("high mannose", "high-mannose"),
    "Complex": ("complex n", "complex-n", "bisected"),
    "Hybrid": ("hybrid",),
    "Paucimannose": ("paucimannose",),
}
N_GLYCAN_KEYWORDS = (
    "n-linked",
    "n-glycan",
    "n glycan",
    "high mannose",
    "paucimannose",
    "hybrid",
    "complex n",
    "bisected",
)
O_GLYCAN_KEYWORDS = (
    "o-linked",
    "o-glycan",
    "o glycan",
    "mucin",
    "o-mannose",
    "o-fucose",
    "o-glucose",
    "o-glcnac",
    "o-galactose",
    "o-galnac",
    "ogalnac",
)
OTHER_MAIN_CLASS_KEYWORDS = (
    "glycosphingolipid",
    "glycolipid",
    "ganglioside",
    "globoside",
    "gpi",
    "glycosaminoglycan",
    "hepar",
    "chondroitin",
    "dermatan",
    "keratan",
    "hyaluron",
)

INLINE_LINKAGE_PATTERN = re.compile(r"[ab?][0-9?]-[0-9?]")
PAREN_LINKAGE_PATTERN = re.compile(r"\([ab?][0-9?]-[0-9?]\)")
PIPE_BRANCH_PATTERN = re.compile(r"\|[0-9?]+")
REDUCING_END_PATTERN = re.compile(r"\+aldi\b")
RESIDUE_PATTERN = re.compile(
    r"[A-Z][A-Za-z0-9,?]{1,20}?"
    r"(?=(?:[ab?][0-9?]-[0-9?])|(?:[ab?](?=[()+\[\]]|$))|(?:[()+\[\]]|$)|(?:\+aldi))"
)
ROOT_ANOMER_PATTERN = re.compile(r"(?<=[A-Za-z0-9,?])[ab?](?=[()+\[\]]|$)")
SIMPLE_MODIFICATION_PATTERN = re.compile(r"(?:\d+[A-Za-z]{1,6}|[A-Za-z]{1,6})")


@dataclass(frozen=True)
class StructureToken:
    """One token from the compact IUPAC structure parser."""

    kind: str
    value: str


@dataclass
class GlycanNode:
    """Simple tree node for one compact IUPAC residue."""

    residue: str
    linkage_to_parent: str = ""
    children: list["GlycanNode"] = field(default_factory=list)


def _load_json_label_list(value: object) -> list[str]:
    """Return one parsed label list from a saved labels-json cell."""

    if isinstance(value, list):
        return [str(label).strip() for label in value if str(label).strip()]
    if value is None:
        return []

    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed_value = json.loads(text_value)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed_value, list):
        return []
    return [str(label).strip() for label in parsed_value if str(label).strip()]


def load_combined_classification_splits(
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
) -> pd.DataFrame:
    """Load the notebook-09 classification tables and attach split names."""

    split_frames: list[pd.DataFrame] = []
    for split_name, split_path in (
        ("train", train_csv_path),
        ("val", val_csv_path),
        ("test", test_csv_path),
    ):
        split_df = pd.read_csv(require_existing_path(split_path, f"{split_name} classification CSV")).copy()
        if LABEL_JSON_COLUMN not in split_df.columns:
            raise ValueError(f"{split_name} classification CSV is missing {LABEL_JSON_COLUMN!r}.")
        split_df[LABEL_LIST_COLUMN] = split_df[LABEL_JSON_COLUMN].map(_load_json_label_list)
        split_df[SPLIT_COLUMN] = split_name
        split_frames.append(split_df)

    combined_df = pd.concat(split_frames, ignore_index=True)
    combined_df[SEQUENCE_COLUMN] = combined_df[SEQUENCE_COLUMN].fillna("").map(str).map(str.strip)
    if combined_df[SEQUENCE_COLUMN].eq("").any():
        raise ValueError("Combined classification dataframe contains blank sequences.")
    return combined_df


def filter_classification_dataframe_by_split(
    classification_df: pd.DataFrame,
    splits_to_include: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Keep only the requested standard split names."""

    if SPLIT_COLUMN not in classification_df.columns:
        raise ValueError(f"classification_df is missing {SPLIT_COLUMN!r}.")

    normalized_splits = [str(split_name).strip().lower() for split_name in (splits_to_include or []) if str(split_name).strip()]
    if not normalized_splits:
        return classification_df.copy().reset_index(drop=True)

    filtered_df = classification_df.loc[
        classification_df[SPLIT_COLUMN].map(lambda value: str(value).strip().lower()).isin(normalized_splits)
    ].copy()
    return filtered_df.reset_index(drop=True)


def _normalize_label_text(label_name: str) -> str:
    text_value = str(label_name).strip().lower().replace("_", " ")
    text_value = text_value.replace("-", " ")
    return re.sub(r"\s+", " ", text_value)


def infer_current_n_o_category(label_values: Sequence[str]) -> str:
    """Collapse the current project labels into an N-vs-O view."""

    normalized_labels = [_normalize_label_text(label_name) for label_name in label_values]
    has_n = any(any(keyword in label_name for keyword in N_GLYCAN_KEYWORDS) for label_name in normalized_labels)
    has_o = any(any(keyword in label_name for keyword in O_GLYCAN_KEYWORDS) for label_name in normalized_labels)

    if has_n and has_o:
        return "Mixed N/O"
    if has_n:
        return "N-glycan"
    if has_o:
        return "O-glycan"
    return "Neither/Other"


def infer_current_main_glycan_class(label_values: Sequence[str]) -> str:
    """Collapse the current project labels into broad glycan classes."""

    n_o_category = infer_current_n_o_category(label_values)
    if n_o_category == "N-glycan":
        return "N-glycan"
    if n_o_category == "O-glycan":
        return "O-glycan"

    normalized_labels = [_normalize_label_text(label_name) for label_name in label_values]
    has_other_main_class = any(
        any(keyword in label_name for keyword in OTHER_MAIN_CLASS_KEYWORDS)
        for label_name in normalized_labels
    )
    if has_other_main_class:
        return "Other glycan"
    return "Other glycan"


def _infer_existing_n_glycan_subclass_matches(label_values: Sequence[str]) -> list[str]:
    """Match the current labels against the notebook-14 subclass keywords."""

    normalized_labels = [_normalize_label_text(label_name) for label_name in label_values if str(label_name).strip()]
    matched_categories: list[str] = []
    for category_name, keywords in N_GLYCAN_SUBCLASS_KEYWORDS.items():
        if any(any(keyword in label_name for keyword in keywords) for label_name in normalized_labels):
            matched_categories.append(category_name)
    return matched_categories


def annotate_current_label_views(classification_df: pd.DataFrame) -> pd.DataFrame:
    """Add the broad current-label views used for structural evaluation."""

    annotated_df = classification_df.copy()
    annotated_df[CURRENT_PRIMARY_SUBTYPE_COLUMN] = annotated_df[LABEL_LIST_COLUMN].map(
        lambda values: sorted(str(value).strip() for value in values if str(value).strip())[0]
        if isinstance(values, list) and values
        else "unlabeled"
    )
    annotated_df[CURRENT_N_O_CATEGORY_COLUMN] = annotated_df[LABEL_LIST_COLUMN].map(infer_current_n_o_category)
    annotated_df[CURRENT_MAIN_CLASS_COLUMN] = annotated_df[LABEL_LIST_COLUMN].map(infer_current_main_glycan_class)
    annotated_df[CURRENT_LABEL_SIGNATURE_COLUMN] = annotated_df[LABEL_LIST_COLUMN].map(
        lambda values: " | ".join(sorted(str(value).strip() for value in values if str(value).strip()))
        if isinstance(values, list)
        else ""
    )
    subclass_matches = annotated_df[LABEL_LIST_COLUMN].map(_infer_existing_n_glycan_subclass_matches)
    annotated_df[CURRENT_SUBCLASS_MATCH_COUNT_COLUMN] = subclass_matches.map(len)
    annotated_df[CURRENT_SUBCLASS_COLUMN] = subclass_matches.map(lambda values: values[0] if len(values) == 1 else "")
    return annotated_df


def _simple_parenthetical_content(glycan_string: str, start_index: int) -> tuple[str, int] | None:
    """Return the contents of a simple non-nested parenthetical group."""

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


def tokenize_compact_iupac_structure(glycan_string: str) -> list[StructureToken]:
    """Parse a compact IUPAC glycan string into lightweight structure tokens."""

    normalized_string = str(glycan_string).strip().replace("[", "(").replace("]", ")")
    tokens: list[StructureToken] = []
    index = 0

    while index < len(normalized_string):
        character = normalized_string[index]
        if character.isspace():
            index += 1
            continue

        match = PAREN_LINKAGE_PATTERN.match(normalized_string, index)
        if match:
            tokens.append(StructureToken("linkage", match.group(0)[1:-1]))
            index = match.end()
            continue

        if character == "(":
            simple_group = _simple_parenthetical_content(normalized_string, index)
            if simple_group is not None:
                content, end_index = simple_group
                if SIMPLE_MODIFICATION_PATTERN.fullmatch(content):
                    index = end_index + 1
                    continue
            tokens.append(StructureToken("branch_open", character))
            index += 1
            continue

        if character == ")":
            tokens.append(StructureToken("branch_close", character))
            index += 1
            continue

        match = REDUCING_END_PATTERN.match(normalized_string, index)
        if match:
            index = match.end()
            continue

        match = PIPE_BRANCH_PATTERN.match(normalized_string, index)
        if match:
            index = match.end()
            continue

        match = INLINE_LINKAGE_PATTERN.match(normalized_string, index)
        if match:
            tokens.append(StructureToken("linkage", match.group(0)))
            index = match.end()
            continue

        match = ROOT_ANOMER_PATTERN.match(normalized_string, index)
        if match:
            tokens.append(StructureToken("root_anomer", match.group(0)))
            index = match.end()
            continue

        match = RESIDUE_PATTERN.match(normalized_string, index)
        if match:
            tokens.append(StructureToken("residue", match.group(0)))
            index = match.end()
            continue

        tokens.append(StructureToken("unknown", character))
        index += 1

    return tokens


def _parse_parenthesized_child(tokens: Sequence[StructureToken], branch_close_index: int) -> tuple[GlycanNode, int]:
    """Parse the subtree inside one parenthesized branch."""

    if branch_close_index - 1 < 0 or tokens[branch_close_index - 1].kind != "linkage":
        raise ValueError("Expected a branch linkage before a closing branch delimiter.")
    linkage_text = tokens[branch_close_index - 1].value
    child_node, next_index = _parse_subtree(
        tokens,
        branch_close_index - 2,
        stop_kinds=frozenset({"branch_open"}),
    )
    child_node.linkage_to_parent = linkage_text
    if next_index < 0 or tokens[next_index].kind != "branch_open":
        raise ValueError("Unmatched branch delimiters in compact IUPAC string.")
    return child_node, next_index - 1


def _parse_subtree(
    tokens: Sequence[StructureToken],
    end_index: int,
    *,
    stop_kinds: frozenset[str] = frozenset(),
) -> tuple[GlycanNode, int]:
    """Parse one compact IUPAC subtree from right to left."""

    index = end_index
    while index >= 0 and tokens[index].kind == "root_anomer":
        index -= 1

    if index < 0 or tokens[index].kind != "residue":
        raise ValueError("Expected a residue while parsing the compact IUPAC structure.")

    root_node = GlycanNode(residue=tokens[index].value)
    index -= 1

    while index >= 0 and tokens[index].kind not in stop_kinds:
        token = tokens[index]
        if token.kind == "branch_close":
            branch_child, index = _parse_parenthesized_child(tokens, index)
            root_node.children.append(branch_child)
            continue

        if token.kind == "linkage":
            donor_child, index = _parse_subtree(
                tokens,
                index - 1,
                stop_kinds=stop_kinds | frozenset({"branch_open"}),
            )
            donor_child.linkage_to_parent = token.value
            root_node.children.append(donor_child)
            continue

        break

    return root_node, index


def parse_compact_iupac_tree(glycan_string: str) -> GlycanNode:
    """Parse one compact IUPAC string into a reducing-end-rooted tree."""

    tokens = tokenize_compact_iupac_structure(glycan_string)
    if not tokens:
        raise ValueError("Compact IUPAC string produced no parseable tokens.")
    if any(token.kind == "unknown" for token in tokens):
        raise ValueError("Compact IUPAC string contains unsupported characters.")

    root_node, next_index = _parse_subtree(tokens, len(tokens) - 1)
    if next_index != -1:
        raise ValueError("Compact IUPAC parser did not consume the full sequence.")
    return root_node


def _residue_family(residue_name: str) -> str:
    """Map one residue token to a coarse residue family."""

    text_value = str(residue_name).strip()
    family_map = (
        ("GlcNAc", "GlcNAc"),
        ("GalNAc", "GalNAc"),
        ("Neu5Gc9Ac", "Neu"),
        ("Neu5,9Ac2", "Neu"),
        ("NeuAc", "Neu"),
        ("NeuGc", "Neu"),
        ("Kdn", "Neu"),
        ("Fuc", "Fuc"),
        ("Xyl", "Xyl"),
        ("Man", "Man"),
        ("Gal", "Gal"),
        ("Glc", "Glc"),
        ("HexNAc", "HexNAc"),
        ("Hex", "Hex"),
    )
    for prefix, family_name in family_map:
        if text_value.startswith(prefix):
            return family_name
    return text_value


def _iter_nodes(root_node: GlycanNode) -> Iterable[GlycanNode]:
    """Yield one node and all of its descendants."""

    yield root_node
    for child_node in root_node.children:
        yield from _iter_nodes(child_node)


def _count_residue_family(root_node: GlycanNode, family_name: str) -> int:
    """Count how many nodes in a subtree match one residue family."""

    return sum(1 for node in _iter_nodes(root_node) if _residue_family(node.residue) == family_name)


def _find_first_child(
    root_node: GlycanNode,
    *,
    residue_family: str,
    linkage_to_parent: str,
) -> GlycanNode | None:
    """Return the first child that matches one family and one linkage."""

    for child_node in root_node.children:
        if _residue_family(child_node.residue) != residue_family:
            continue
        if str(child_node.linkage_to_parent) != str(linkage_to_parent):
            continue
        return child_node
    return None


def _looks_like_partial_n_core(root_node: GlycanNode) -> bool:
    """Return True for obvious N-glycan chitobiose/core-trimmed patterns."""

    if _residue_family(root_node.residue) != "GlcNAc":
        return False
    second_glcnac = _find_first_child(root_node, residue_family="GlcNAc", linkage_to_parent="b1-4")
    if second_glcnac is None:
        return False
    central_man = _find_first_child(second_glcnac, residue_family="Man", linkage_to_parent="b1-4")
    return central_man is not None


def _extract_full_n_core(root_node: GlycanNode) -> dict[str, GlycanNode] | None:
    """Return the main N-glycan core residues when the canonical core is present."""

    if _residue_family(root_node.residue) != "GlcNAc":
        return None
    second_glcnac = _find_first_child(root_node, residue_family="GlcNAc", linkage_to_parent="b1-4")
    if second_glcnac is None:
        return None
    central_man = _find_first_child(second_glcnac, residue_family="Man", linkage_to_parent="b1-4")
    if central_man is None:
        return None

    arm_a1_3 = _find_first_child(central_man, residue_family="Man", linkage_to_parent="a1-3")
    arm_a1_6 = _find_first_child(central_man, residue_family="Man", linkage_to_parent="a1-6")
    if arm_a1_3 is None or arm_a1_6 is None:
        return None

    return {
        "reducing_glcnac": root_node,
        "chitobiose_glcnac": second_glcnac,
        "central_man": central_man,
        "arm_a1_3": arm_a1_3,
        "arm_a1_6": arm_a1_6,
    }


def _analyze_arm(arm_root: GlycanNode) -> dict[str, Any]:
    """Summarize one alpha1-3 or alpha1-6 arm above the core mannose."""

    descendants = list(_iter_nodes(arm_root))
    descendant_families = [_residue_family(node.residue) for node in descendants[1:]]
    direct_child_families = [_residue_family(child_node.residue) for child_node in arm_root.children]
    direct_glcnac_children = [
        child_node
        for child_node in arm_root.children
        if _residue_family(child_node.residue) == "GlcNAc"
    ]

    has_direct_glcnac = bool(direct_glcnac_children)
    has_any_non_man_descendant = any(family_name != "Man" for family_name in descendant_families)
    mannose_only = bool(descendant_families) and not has_any_non_man_descendant
    bare_core_arm = not descendant_families

    if has_direct_glcnac:
        status = "glcnac_processed"
    elif bare_core_arm:
        status = "bare_core_arm"
    elif mannose_only:
        status = "mannose_only_extension"
    else:
        status = "non_mannose_without_direct_glcnac"

    return {
        "status": status,
        "has_direct_glcnac": has_direct_glcnac,
        "mannose_only": mannose_only,
        "bare_core_arm": bare_core_arm,
        "descendant_families": " | ".join(descendant_families),
        "direct_child_families": " | ".join(direct_child_families),
    }


def classify_n_glycan_sequence(sequence: str) -> dict[str, Any]:
    """Classify one compact IUPAC sequence by structural N-glycan rules."""

    cleaned_sequence = str(sequence).strip()
    if not cleaned_sequence:
        return {
            STRUCTURAL_CLASS_COLUMN: "n_glycan_unresolved",
            STRUCTURAL_BINARY_COLUMN: pd.NA,
            STRUCTURAL_SUBCLASS_COLUMN: "Unresolved N-glycan",
            STRUCTURAL_REASON_COLUMN: "blank_sequence",
            STRUCTURAL_CONFIDENCE_COLUMN: "low",
            STRUCTURAL_CORE_STATUS_COLUMN: "blank_sequence",
            STRUCTURAL_ARM3_STATUS_COLUMN: "",
            STRUCTURAL_ARM6_STATUS_COLUMN: "",
            STRUCTURAL_TOTAL_MAN_COLUMN: pd.NA,
            STRUCTURAL_PARSE_STATUS_COLUMN: "blank_sequence",
        }

    try:
        parsed_tree = parse_compact_iupac_tree(cleaned_sequence)
    except ValueError as error:
        return {
            STRUCTURAL_CLASS_COLUMN: "n_glycan_unresolved",
            STRUCTURAL_BINARY_COLUMN: pd.NA,
            STRUCTURAL_SUBCLASS_COLUMN: "Unresolved N-glycan",
            STRUCTURAL_REASON_COLUMN: str(error),
            STRUCTURAL_CONFIDENCE_COLUMN: "low",
            STRUCTURAL_CORE_STATUS_COLUMN: "parse_failed",
            STRUCTURAL_ARM3_STATUS_COLUMN: "",
            STRUCTURAL_ARM6_STATUS_COLUMN: "",
            STRUCTURAL_TOTAL_MAN_COLUMN: pd.NA,
            STRUCTURAL_PARSE_STATUS_COLUMN: "parse_failed",
        }

    full_core = _extract_full_n_core(parsed_tree)
    if full_core is None:
        if _looks_like_partial_n_core(parsed_tree):
            total_mannose = _count_residue_family(parsed_tree, "Man")
            return {
                STRUCTURAL_CLASS_COLUMN: "n_glycan_paucimannose_or_truncated",
                STRUCTURAL_BINARY_COLUMN: True,
                STRUCTURAL_SUBCLASS_COLUMN: "Paucimannose/truncated",
                STRUCTURAL_REASON_COLUMN: "partial_n_glycan_core",
                STRUCTURAL_CONFIDENCE_COLUMN: "medium",
                STRUCTURAL_CORE_STATUS_COLUMN: "partial_core",
                STRUCTURAL_ARM3_STATUS_COLUMN: "",
                STRUCTURAL_ARM6_STATUS_COLUMN: "",
                STRUCTURAL_TOTAL_MAN_COLUMN: total_mannose,
                STRUCTURAL_PARSE_STATUS_COLUMN: "parsed",
            }

        return {
            STRUCTURAL_CLASS_COLUMN: "not_n_glycan",
            STRUCTURAL_BINARY_COLUMN: False,
            STRUCTURAL_SUBCLASS_COLUMN: "Not N-glycan",
            STRUCTURAL_REASON_COLUMN: "no_n_glycan_core",
            STRUCTURAL_CONFIDENCE_COLUMN: "high",
            STRUCTURAL_CORE_STATUS_COLUMN: "core_not_found",
            STRUCTURAL_ARM3_STATUS_COLUMN: "",
            STRUCTURAL_ARM6_STATUS_COLUMN: "",
            STRUCTURAL_TOTAL_MAN_COLUMN: _count_residue_family(parsed_tree, "Man"),
            STRUCTURAL_PARSE_STATUS_COLUMN: "parsed",
        }

    arm3_analysis = _analyze_arm(full_core["arm_a1_3"])
    arm6_analysis = _analyze_arm(full_core["arm_a1_6"])
    total_mannose = _count_residue_family(parsed_tree, "Man")

    if arm3_analysis["has_direct_glcnac"] and arm6_analysis["has_direct_glcnac"]:
        structural_class = "n_glycan_complex"
        structural_subclass = "Complex"
        reason = "both_core_arms_glcnac_processed"
        confidence = "high"
    elif arm3_analysis["has_direct_glcnac"] ^ arm6_analysis["has_direct_glcnac"]:
        other_arm = arm6_analysis if arm3_analysis["has_direct_glcnac"] else arm3_analysis
        if other_arm["status"] in {"mannose_only_extension", "bare_core_arm"}:
            structural_class = "n_glycan_hybrid"
            structural_subclass = "Hybrid"
            reason = "one_core_arm_glcnac_processed"
            confidence = "high"
        else:
            structural_class = "n_glycan_unresolved"
            structural_subclass = "Unresolved N-glycan"
            reason = "mixed_arm_signal"
            confidence = "medium"
    elif arm3_analysis["status"] in {"mannose_only_extension", "bare_core_arm"} and arm6_analysis["status"] in {
        "mannose_only_extension",
        "bare_core_arm",
    }:
        if total_mannose >= 5:
            structural_class = "n_glycan_high_mannose"
            structural_subclass = "High mannose"
            reason = "mannose_only_branches_with_extended_man_count"
            confidence = "high"
        else:
            structural_class = "n_glycan_paucimannose_or_truncated"
            structural_subclass = "Paucimannose/truncated"
            reason = "full_core_but_low_man_count"
            confidence = "medium"
    else:
        structural_class = "n_glycan_unresolved"
        structural_subclass = "Unresolved N-glycan"
        reason = "full_core_but_arm_rules_inconclusive"
        confidence = "medium"

    return {
        STRUCTURAL_CLASS_COLUMN: structural_class,
        STRUCTURAL_BINARY_COLUMN: structural_class != "not_n_glycan",
        STRUCTURAL_SUBCLASS_COLUMN: structural_subclass,
        STRUCTURAL_REASON_COLUMN: reason,
        STRUCTURAL_CONFIDENCE_COLUMN: confidence,
        STRUCTURAL_CORE_STATUS_COLUMN: "full_core",
        STRUCTURAL_ARM3_STATUS_COLUMN: arm3_analysis["status"],
        STRUCTURAL_ARM6_STATUS_COLUMN: arm6_analysis["status"],
        STRUCTURAL_TOTAL_MAN_COLUMN: total_mannose,
        STRUCTURAL_PARSE_STATUS_COLUMN: "parsed",
    }


def annotate_structural_n_glycan_classes(
    classification_df: pd.DataFrame,
    *,
    sequence_column: str = SEQUENCE_COLUMN,
) -> pd.DataFrame:
    """Apply the structural classifier to every row in a dataframe."""

    if sequence_column not in classification_df.columns:
        raise ValueError(f"classification_df is missing {sequence_column!r}.")

    annotated_df = classification_df.copy().reset_index(drop=True)
    classification_rows = annotated_df[sequence_column].map(classify_n_glycan_sequence).tolist()
    structural_df = pd.DataFrame(classification_rows)
    return pd.concat([annotated_df, structural_df], axis=1)


def summarize_structural_classes_by_split(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize structural classes by split."""

    summary_df = (
        annotated_df.groupby([SPLIT_COLUMN, STRUCTURAL_CLASS_COLUMN], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    summary_df[STRUCTURAL_CLASS_COLUMN] = pd.Categorical(
        summary_df[STRUCTURAL_CLASS_COLUMN],
        categories=STRUCTURAL_CLASS_ORDER,
        ordered=True,
    )
    return summary_df.sort_values([SPLIT_COLUMN, STRUCTURAL_CLASS_COLUMN], kind="stable").reset_index(drop=True)


def summarize_structural_reasons(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize structural assignment reasons and confidence levels."""

    summary_df = (
        annotated_df.groupby(
            [STRUCTURAL_CLASS_COLUMN, STRUCTURAL_REASON_COLUMN, STRUCTURAL_CONFIDENCE_COLUMN],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    summary_df[STRUCTURAL_CLASS_COLUMN] = pd.Categorical(
        summary_df[STRUCTURAL_CLASS_COLUMN],
        categories=STRUCTURAL_CLASS_ORDER,
        ordered=True,
    )
    return summary_df.sort_values(
        [STRUCTURAL_CLASS_COLUMN, "count", STRUCTURAL_REASON_COLUMN],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_binary_agreement_table(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Compare structural N-glycan calls against the current broad labels."""

    agreement_df = annotated_df.copy()
    agreement_df["current_is_n_glycan"] = agreement_df[CURRENT_MAIN_CLASS_COLUMN].map(str).eq("N-glycan")
    agreement_df["structural_is_n_glycan_text"] = agreement_df[STRUCTURAL_BINARY_COLUMN].map(
        lambda value: "unresolved" if pd.isna(value) else ("yes" if bool(value) else "no")
    )
    summary_df = (
        agreement_df.groupby(
            ["current_is_n_glycan", "structural_is_n_glycan_text", SPLIT_COLUMN],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    return summary_df.sort_values(
        ["current_is_n_glycan", "structural_is_n_glycan_text", SPLIT_COLUMN],
        kind="stable",
    ).reset_index(drop=True)


def build_subclass_agreement_table(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Compare structural subclass calls against the current label-derived subclass view."""

    subclass_df = annotated_df.loc[
        annotated_df[CURRENT_MAIN_CLASS_COLUMN].map(str).eq("N-glycan")
    ].copy()
    subclass_df["current_subclass_display"] = subclass_df[CURRENT_SUBCLASS_COLUMN].where(
        subclass_df[CURRENT_SUBCLASS_COLUMN].fillna("").map(str).ne(""),
        "No single current subclass label",
    )
    summary_df = (
        subclass_df.groupby(
            [SPLIT_COLUMN, "current_subclass_display", STRUCTURAL_SUBCLASS_COLUMN],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    summary_df[STRUCTURAL_SUBCLASS_COLUMN] = pd.Categorical(
        summary_df[STRUCTURAL_SUBCLASS_COLUMN],
        categories=STRUCTURAL_SUBCLASS_ORDER,
        ordered=True,
    )
    return summary_df.sort_values(
        [SPLIT_COLUMN, "current_subclass_display", STRUCTURAL_SUBCLASS_COLUMN],
        kind="stable",
    ).reset_index(drop=True)


def build_structural_evaluation_examples(
    annotated_df: pd.DataFrame,
    *,
    disagreement_limit: int = 200,
    unresolved_limit: int = 200,
) -> dict[str, pd.DataFrame]:
    """Return example tables for manual review."""

    comparison_df = annotated_df.copy()
    comparison_df["current_is_n_glycan"] = comparison_df[CURRENT_MAIN_CLASS_COLUMN].map(str).eq("N-glycan")
    comparison_df["structural_binary_disagrees_with_current"] = comparison_df[STRUCTURAL_BINARY_COLUMN].map(
        lambda value: False if pd.isna(value) else bool(value)
    ) != comparison_df["current_is_n_glycan"]

    disagreement_examples_df = comparison_df.loc[
        comparison_df["structural_binary_disagrees_with_current"]
    ].copy()
    disagreement_examples_df = disagreement_examples_df.sort_values(
        [SPLIT_COLUMN, STRUCTURAL_CLASS_COLUMN, ACCESSION_COLUMN],
        kind="stable",
    ).head(int(disagreement_limit))

    unresolved_examples_df = comparison_df.loc[
        comparison_df[STRUCTURAL_CLASS_COLUMN].eq("n_glycan_unresolved")
    ].copy()
    unresolved_examples_df = unresolved_examples_df.sort_values(
        [SPLIT_COLUMN, STRUCTURAL_REASON_COLUMN, ACCESSION_COLUMN],
        kind="stable",
    ).head(int(unresolved_limit))

    selected_columns = [
        SPLIT_COLUMN,
        ACCESSION_COLUMN,
        SEQUENCE_COLUMN,
        CURRENT_MAIN_CLASS_COLUMN,
        CURRENT_SUBCLASS_COLUMN,
        CURRENT_LABEL_SIGNATURE_COLUMN,
        STRUCTURAL_CLASS_COLUMN,
        STRUCTURAL_SUBCLASS_COLUMN,
        STRUCTURAL_REASON_COLUMN,
        STRUCTURAL_CONFIDENCE_COLUMN,
        STRUCTURAL_CORE_STATUS_COLUMN,
        STRUCTURAL_ARM3_STATUS_COLUMN,
        STRUCTURAL_ARM6_STATUS_COLUMN,
        STRUCTURAL_TOTAL_MAN_COLUMN,
    ]
    return {
        "disagreement_examples_df": disagreement_examples_df.loc[:, [column for column in selected_columns if column in disagreement_examples_df.columns]].copy(),
        "unresolved_examples_df": unresolved_examples_df.loc[:, [column for column in selected_columns if column in unresolved_examples_df.columns]].copy(),
    }


def build_structural_classification_output_paths(
    project_root: str | Path,
    *,
    run_label: str,
) -> dict[str, Path]:
    """Return the standard output paths for the structural-classification workflow."""

    run_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_label).strip()).strip("_") or "run"
    results_dir = Path(project_root) / "results" / "n_glycan_structural_classification" / run_slug
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "run_config_path": results_dir / "run_config.json",
        "annotated_rows_path": results_dir / "annotated_structural_classification.csv",
        "class_summary_path": results_dir / "structural_class_summary_by_split.csv",
        "reason_summary_path": results_dir / "structural_reason_summary.csv",
        "binary_agreement_path": results_dir / "binary_agreement_summary.csv",
        "subclass_agreement_path": results_dir / "subclass_agreement_summary.csv",
        "disagreement_examples_path": results_dir / "binary_disagreement_examples.csv",
        "unresolved_examples_path": results_dir / "unresolved_examples.csv",
    }


def build_structural_classification_run_config(
    *,
    project_root: str | Path,
    classification_prep_dir: str | Path,
    splits_to_include: Sequence[str],
    structural_run_label: str,
    overwrite_existing_outputs: bool,
    disagreement_limit: int,
    unresolved_limit: int,
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the saved config payload for one structural-classification run."""

    return {
        "project_root": str(project_root),
        "classification_prep_dir": str(classification_prep_dir),
        "splits_to_include": [str(split_name) for split_name in splits_to_include],
        "structural_run_label": str(structural_run_label),
        "overwrite_existing_outputs": bool(overwrite_existing_outputs),
        "disagreement_limit": int(disagreement_limit),
        "unresolved_limit": int(unresolved_limit),
        "output_paths": stringify_path_values(dict(output_paths)),
    }


def save_structural_classification_run_config(
    output_path: str | Path,
    config_payload: Mapping[str, Any],
) -> Path:
    """Write one pretty JSON config payload for the structural workflow."""

    return write_json(output_path, stringify_path_values(dict(config_payload)))


def run_structural_classification_workflow(
    *,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    output_paths: Mapping[str, Path],
    splits_to_include: Sequence[str] | None = None,
    overwrite_existing_outputs: bool = True,
    disagreement_limit: int = 200,
    unresolved_limit: int = 200,
) -> dict[str, Any]:
    """Run the compact-IUPAC structural classifier end to end."""

    output_paths = {key: Path(value) for key, value in output_paths.items()}
    validate_output_paths(
        {
            key: value
            for key, value in output_paths.items()
            if key not in {"results_dir", "run_config_path"}
        },
        overwrite_existing_outputs=overwrite_existing_outputs,
    )

    combined_df = load_combined_classification_splits(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
    )
    filtered_df = filter_classification_dataframe_by_split(
        combined_df,
        splits_to_include=splits_to_include,
    )
    current_label_df = annotate_current_label_views(filtered_df)
    annotated_df = annotate_structural_n_glycan_classes(current_label_df)

    class_summary_df = summarize_structural_classes_by_split(annotated_df)
    reason_summary_df = summarize_structural_reasons(annotated_df)
    binary_agreement_df = build_binary_agreement_table(annotated_df)
    subclass_agreement_df = build_subclass_agreement_table(annotated_df)
    example_tables = build_structural_evaluation_examples(
        annotated_df,
        disagreement_limit=disagreement_limit,
        unresolved_limit=unresolved_limit,
    )

    annotated_df.to_csv(output_paths["annotated_rows_path"], index=False)
    class_summary_df.to_csv(output_paths["class_summary_path"], index=False)
    reason_summary_df.to_csv(output_paths["reason_summary_path"], index=False)
    binary_agreement_df.to_csv(output_paths["binary_agreement_path"], index=False)
    subclass_agreement_df.to_csv(output_paths["subclass_agreement_path"], index=False)
    example_tables["disagreement_examples_df"].to_csv(output_paths["disagreement_examples_path"], index=False)
    example_tables["unresolved_examples_df"].to_csv(output_paths["unresolved_examples_path"], index=False)

    return {
        "annotated_df": annotated_df,
        "class_summary_df": class_summary_df,
        "reason_summary_df": reason_summary_df,
        "binary_agreement_df": binary_agreement_df,
        "subclass_agreement_df": subclass_agreement_df,
        "disagreement_examples_df": example_tables["disagreement_examples_df"],
        "unresolved_examples_df": example_tables["unresolved_examples_df"],
        "output_paths": dict(output_paths),
    }
