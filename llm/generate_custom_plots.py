#!/usr/bin/env python3
"""
Utility script to generate enrichment (PSEA) and volcano plots directly
from the CSV outputs stored in `llm/perturbation_outputs`.

It inspects each perturbation directory (e.g., `gene_CHCHD2`, `drug_*`),
loads the pathway analysis CSV files for GO / Reactome / KEGG (RNA and protein),
and saves horizontal bar plots that summarize the top pathways. It also
creates volcano plots for both RNA and protein differential expression CSVs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate enrichment and volcano plots from CSV outputs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("llm/perturbation_outputs"),
        help="Root directory containing perturbation subfolders.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=20,
        help="Maximum number of pathways to display per enrichment plot.",
    )
    parser.add_argument(
        "--pval-column",
        default="Adjusted P-value",
        help="Column to use for significance (-log10) in enrichment CSVs.",
    )
    parser.add_argument(
        "--sig-threshold",
        type=float,
        default=0.05,
        help="P-value threshold used for volcano plot highlighting.",
    )
    parser.add_argument(
        "--fc-threshold",
        type=float,
        default=1.0,
        help="Absolute log2 fold-change threshold for volcano plot highlighting.",
    )
    parser.add_argument(
        "--plot-subdir",
        default="custom_plots",
        help="Subdirectory (under each perturbation) where plots will be saved.",
    )
    return parser.parse_args()


def discover_perturbation_dirs(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir() and (sub.name.startswith("gene_") or sub.name.startswith("drug_")):
            yield sub


def select_pathway_files(pathway_dir: Path, prefix: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (rna_file, protein_file) for a given prefix (go_enrichment, reactome_enrichment, kegg_enrichment).
    """
    rna_candidates = sorted(pathway_dir.glob(f"{prefix}_rna*.csv"))
    protein_candidates = sorted(
        f for f in pathway_dir.glob(f"{prefix}*.csv") if "_rna" not in f.stem
    )
    return (rna_candidates[0] if rna_candidates else None, protein_candidates[0] if protein_candidates else None)


def plot_enrichment(
    csv_path: Path,
    title: str,
    output_path: Path,
    max_terms: int,
    pval_column: str,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or "Term" not in df.columns:
        print(f"  ⚠️  Skipping {csv_path.name}: required columns missing or file empty.")
        return

    df = df.copy()
    if "NES" not in df.columns:
        print(f"  ⚠️  No NES column in {csv_path.name}; skipping plot.")
        return

    top_terms = (
        df.assign(abs_nes=lambda d: d["NES"].abs())
        .sort_values(by="abs_nes", ascending=False)
        .head(max_terms)
        .sort_values(by="NES")
    )

    colors = top_terms["NES"].apply(lambda x: "#1b9e77" if x >= 0 else "#d95f02")

    plt.figure(figsize=(10, max(4, 0.35 * len(top_terms))))
    sns.barplot(
        data=top_terms,
        x="NES",
        y="Term",
        palette=colors,
        orient="h",
    )
    plt.title(title)
    plt.xlabel("Normalized Enrichment Score (NES)")
    plt.ylabel("")
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.xlim(
        min(0, top_terms["NES"].min()) * 1.1,
        max(0, top_terms["NES"].max()) * 1.1,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"    ✓ Saved {output_path.relative_to(output_path.parent.parent)}")


def plot_volcano(
    csv_path: Path,
    title: str,
    output_path: Path,
    sig_threshold: float,
    fc_threshold: float,
) -> None:
    df = pd.read_csv(csv_path)
    if "gene" not in df.columns or "log2fc" not in df.columns:
        print(f"  ⚠️  Missing columns for volcano plot in {csv_path.name}; skipping.")
        return

    pval_col = None
    for candidate in ("pvalue", "pval", "P-value", "p_value"):
        if candidate in df.columns:
            pval_col = candidate
            break
    if not pval_col:
        print(f"  ⚠️  No p-value column found in {csv_path.name}; skipping volcano plot.")
        return

    df = df.dropna(subset=["log2fc", pval_col]).copy()
    if df.empty:
        print(f"  ⚠️  No valid rows in {csv_path.name}; skipping volcano plot.")
        return

    df["-log10(p)"] = df[pval_col].apply(lambda p: -math.log10(p) if p > 0 else float("inf"))
    df["significant"] = (df[pval_col] <= sig_threshold) & (df["log2fc"].abs() >= fc_threshold)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="log2fc",
        y="-log10(p)",
        hue="significant",
        palette={True: "#d73027", False: "#8da0cb"},
        edgecolor=None,
        s=12,
        legend=False,
    )
    plt.axhline(-math.log10(sig_threshold), color="#999999", linestyle="--", linewidth=0.8)
    plt.axvline(fc_threshold, color="#999999", linestyle="--", linewidth=0.8)
    plt.axvline(-fc_threshold, color="#999999", linestyle="--", linewidth=0.8)
    plt.title(title)
    plt.xlabel("log2 fold-change")
    plt.ylabel("-log10(p-value)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"    ✓ Saved {output_path.relative_to(output_path.parent.parent)}")


def process_perturbation(
    perturbation_dir: Path,
    max_terms: int,
    pval_column: str,
    sig_threshold: float,
    fc_threshold: float,
    plot_subdir: str,
) -> None:
    pathway_dir = perturbation_dir / "pathway_analysis"
    if not pathway_dir.exists():
        print(f"  ⚠️  No pathway_analysis directory in {perturbation_dir.name}; skipping.")
        return

    plot_root = perturbation_dir / plot_subdir

    enrichment_prefixes = {
        "go_enrichment": "GO",
        "reactome_enrichment": "Reactome",
        "kegg_enrichment": "KEGG",
    }

    print(f"\n== {perturbation_dir.name} ==")

    for prefix, label in enrichment_prefixes.items():
        rna_file, protein_file = select_pathway_files(pathway_dir, prefix)
        if rna_file:
            plot_enrichment(
                rna_file,
                f"{label} pathways (RNA) – {perturbation_dir.name}",
                plot_root / f"{label.lower()}_rna_psea.png",
                max_terms,
                pval_column,
            )
        if protein_file:
            plot_enrichment(
                protein_file,
                f"{label} pathways (protein) – {perturbation_dir.name}",
                plot_root / f"{label.lower()}_protein_psea.png",
                max_terms,
                pval_column,
            )

    diff_expr_files = {
        "rna": sorted(pathway_dir.glob("differential_expression_rna*.csv")),
        "protein": sorted(pathway_dir.glob("differential_expression_protein*.csv")),
    }
    top_protein_file = perturbation_dir / "top_proteins.csv"

    if diff_expr_files["rna"]:
        plot_volcano(
            diff_expr_files["rna"][0],
            f"RNA differential expression – {perturbation_dir.name}",
            plot_root / "volcano_rna.png",
            sig_threshold,
            fc_threshold,
        )

    protein_source = None
    if top_protein_file.exists():
        protein_source = top_protein_file
    elif diff_expr_files["protein"]:
        protein_source = diff_expr_files["protein"][0]

    if protein_source:
        plot_volcano(
            protein_source,
            f"Protein differential expression – {perturbation_dir.name}",
            plot_root / "volcano_protein.png",
            sig_threshold,
            fc_threshold,
        )


def main() -> None:
    args = parse_args()
    sns.set_style("whitegrid")

    perturbation_dirs = list(discover_perturbation_dirs(args.base_dir))
    if not perturbation_dirs:
        print(f"No perturbation directories found under {args.base_dir}.")
        return

    print(f"Found {len(perturbation_dirs)} perturbation directories.")
    for pert_dir in perturbation_dirs:
        process_perturbation(
            perturbation_dir=pert_dir,
            max_terms=args.max_terms,
            pval_column=args.pval_column,
            sig_threshold=args.sig_threshold,
            fc_threshold=args.fc_threshold,
            plot_subdir=args.plot_subdir,
        )


if __name__ == "__main__":
    main()

