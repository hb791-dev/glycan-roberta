"""Compatibility import surface for similarity helpers.

The implementation now lives in smaller modules grouped by responsibility, but
this shim keeps the original notebook-facing import path stable.
"""

from src.similarity_core import (
    _effective_max_length,
    _get_encoder,
    _image_path_to_data_uri,
    build_tokenization_preview,
    build_variant_preview_sequences,
    collect_preview_sequences,
    compare_sequence_pair,
    compare_sequence_pairs,
    embed_sequences,
    load_similarity_artifacts,
    plot_similarity_heatmap,
    resolve_device,
    run_similarity_analysis,
    save_similarity_outputs,
    similarity_matrix,
    similarity_matrix_dataframe,
    tokenize_sequence,
    validate_similarity_inputs,
)
from src.similarity_scaleup import (
    _build_matrix_labels,
    _clean_similarity_dataframe,
    _collect_scaleup_html_sequences,
    _format_summary_table_rows,
    _summarize_similarity_values,
    build_all_vs_all_artifacts,
    build_embedding_lookup_for_dataframe,
    build_similarity_distribution_summary,
    build_threshold_cloud_table,
    compare_queries_to_corpus,
    plot_similarity_distribution_histogram,
    render_scaleup_index_html,
    render_specific_vs_all_html,
    run_scaleup_similarity_analysis,
    save_scaleup_similarity_outputs,
    validate_scaleup_similarity_inputs,
)
from src.similarity_variants import (
    VARIANT_SET_ORDER,
    _humanize_label,
    _normalize_variant_set_name,
    _sort_variant_results,
    _variant_set_heading,
    _variant_set_sort_key,
    build_anchor_similarity_matrices,
    build_variant_summary_tables,
    compare_anchor_variants,
    plot_variant_similarity_histogram,
    render_anchor_similarity_html,
    render_variant_index_html,
    run_variant_similarity_analysis,
    save_variant_similarity_outputs,
    validate_variant_similarity_inputs,
)

__all__ = [name for name in globals() if not name.startswith('__')]
