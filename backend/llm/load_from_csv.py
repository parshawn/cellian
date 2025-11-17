"""Helper script to load data from CSV files and convert to payload format."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_deg_list_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load DEG list from CSV file.
    
    Expected CSV columns:
    - gene: Gene symbol
    - log2fc: Log2 fold change
    - pval: P-value
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of DEG dicts
    """
    if PANDAS_AVAILABLE:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
        
        gene_col = None
        for col in ['gene', 'genes', 'gene_symbol', 'symbol', 'gene_id']:
            if col in df.columns:
                gene_col = col
                break
        
        log2fc_col = None
        for col in ['log2fc', 'log2_fc', 'logfc', 'fold_change', 'fc']:
            if col in df.columns:
                log2fc_col = col
                break
        
        pval_col = None
        for col in ['pval', 'p_value', 'pvalue', 'p', 'adj_pval', 'fdr']:
            if col in df.columns:
                pval_col = col
                break
        
        if not gene_col:
            raise ValueError(f"Could not find gene column. Available: {df.columns.tolist()}")
        
        deg_list = []
        for _, row in df.iterrows():
            deg = {
                "gene": str(row[gene_col]).strip(),
                "log2fc": float(row[log2fc_col]) if log2fc_col and log2fc_col in row else 0.0,
                "pval": float(row[pval_col]) if pval_col and pval_col in row else 1.0
            }
            deg_list.append(deg)
        
        return deg_list
    else:
        # Use built-in csv module
        deg_list = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize keys
                row_lower = {k.lower().strip(): v for k, v in row.items()}
                
                gene_col = None
                for col in ['gene', 'genes', 'gene_symbol', 'symbol', 'gene_id']:
                    if col in row_lower:
                        gene_col = col
                        break
                
                log2fc_col = None
                for col in ['log2fc', 'log2_fc', 'logfc', 'fold_change', 'fc']:
                    if col in row_lower:
                        log2fc_col = col
                        break
                
                pval_col = None
                for col in ['pval', 'p_value', 'pvalue', 'p', 'adj_pval', 'fdr']:
                    if col in row_lower:
                        pval_col = col
                        break
                
                if not gene_col or not row_lower.get(gene_col):
                    continue
                
                try:
                    deg = {
                        "gene": str(row_lower[gene_col]).strip(),
                        "log2fc": float(row_lower[log2fc_col]) if log2fc_col and row_lower.get(log2fc_col) else 0.0,
                        "pval": float(row_lower[pval_col]) if pval_col and row_lower.get(pval_col) else 1.0
                    }
                    deg_list.append(deg)
                except (ValueError, TypeError):
                    continue
        
        if not deg_list:
            raise ValueError(f"Could not parse any DEGs from CSV. Check column names.")
        
        return deg_list


def load_pathways_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load pathway enrichment results from CSV.
    
    Expected CSV columns:
    - id or pathway_id: Pathway identifier
    - name or pathway_name: Pathway name
    - NES: Normalized enrichment score
    - FDR: False discovery rate
    - member_genes or genes: Comma-separated gene list (optional)
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of pathway dicts
    """
    if PANDAS_AVAILABLE:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
        
        id_col = None
        for col in ['id', 'pathway_id', 'pathway', 'term']:
            if col in df.columns:
                id_col = col
                break
        
        name_col = None
        for col in ['name', 'pathway_name', 'description', 'term_name']:
            if col in df.columns:
                name_col = col
                break
        
        nes_col = None
        for col in ['nes', 'normalized_enrichment_score', 'enrichment_score']:
            if col in df.columns:
                nes_col = col
                break
        
        fdr_col = None
        for col in ['fdr', 'adj_pval', 'padj', 'q_value', 'qvalue']:
            if col in df.columns:
                fdr_col = col
                break
        
        genes_col = None
        for col in ['member_genes', 'genes', 'gene_list', 'leading_edge']:
            if col in df.columns:
                genes_col = col
                break
        
        if not id_col:
            raise ValueError(f"Could not find pathway ID column. Available: {df.columns.tolist()}")
        
        pathways = []
        for _, row in df.iterrows():
            pathway = {
                "id": str(row[id_col]).strip(),
                "name": str(row[name_col]).strip() if name_col and name_col in row else str(row[id_col]),
                "source": "GSEA",
                "NES": float(row[nes_col]) if nes_col and nes_col in row else 0.0,
                "FDR": float(row[fdr_col]) if fdr_col and fdr_col in row else 1.0,
                "member_genes": []
            }
            
            if genes_col and genes_col in row:
                genes_str = str(row[genes_col])
                if genes_str and genes_str != 'nan':
                    if ',' in genes_str:
                        pathway["member_genes"] = [g.strip() for g in genes_str.split(',')]
                    else:
                        pathway["member_genes"] = [g.strip() for g in genes_str.split()]
            
            pathways.append(pathway)
        
        return pathways
    else:
        # Use built-in csv module
        pathways = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                row_lower = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    key = k.lower().strip() if k else ""
                    value = v if v else ""
                    row_lower[key] = value
                
                id_col = None
                for col in ['id', 'pathway_id', 'pathway', 'term']:
                    if col in row_lower:
                        id_col = col
                        break
                
                name_col = None
                for col in ['name', 'pathway_name', 'description', 'term_name']:
                    if col in row_lower:
                        name_col = col
                        break
                
                nes_col = None
                for col in ['nes', 'normalized_enrichment_score', 'enrichment_score']:
                    if col in row_lower:
                        nes_col = col
                        break
                
                fdr_col = None
                for col in ['fdr', 'adj_pval', 'padj', 'q_value', 'qvalue']:
                    if col in row_lower:
                        fdr_col = col
                        break
                
                genes_col = None
                for col in ['member_genes', 'genes', 'gene_list', 'leading_edge']:
                    if col in row_lower:
                        genes_col = col
                        break
                
                if not id_col or not row_lower.get(id_col):
                    continue
                
                try:
                    pathway = {
                        "id": str(row_lower[id_col]).strip(),
                        "name": str(row_lower[name_col]).strip() if name_col and row_lower.get(name_col) else str(row_lower[id_col]),
                        "source": "GSEA",
                        "NES": float(row_lower[nes_col]) if nes_col and row_lower.get(nes_col) else 0.0,
                        "FDR": float(row_lower[fdr_col]) if fdr_col and row_lower.get(fdr_col) else 1.0,
                        "member_genes": []
                    }
                    
                    if genes_col and row_lower.get(genes_col):
                        genes_str = str(row_lower[genes_col])
                        if genes_str and genes_str.lower() != 'nan':
                            if ',' in genes_str:
                                pathway["member_genes"] = [g.strip() for g in genes_str.split(',') if g.strip()]
                            else:
                                pathway["member_genes"] = [g.strip() for g in genes_str.split() if g.strip()]
                    
                    pathways.append(pathway)
                except (ValueError, TypeError):
                    continue
        
        if not pathways:
            raise ValueError(f"Could not parse any pathways from CSV. Check column names.")
        
        return pathways


def create_payload_from_files(
    deg_csv: Optional[str] = None,
    pathways_csv: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    validated_edges: Optional[List[Dict[str, Any]]] = None,
    phenotypes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create payload from CSV files and optional data.
    
    Args:
        deg_csv: Path to DEG CSV file
        pathways_csv: Path to pathways CSV file
        context: Context dict (perturbation, cell_type, species, user_question)
        validated_edges: List of validated edges (optional)
        phenotypes: List of phenotypes (optional, will be computed if not provided)
    
    Returns:
        Payload dict matching expected schema
    """
    payload = {
        "context": context or {
            "perturbation": "Unknown",
            "cell_type": "",
            "species": "human",
            "user_question": ""
        },
        "validated_edges": validated_edges or [],
        "deg_list": [],
        "pathways": [],
        "phenotypes": phenotypes or [],
        "evidence": {
            "datasets": [],
            "papers": []
        }
    }
    
    # Load DEGs
    if deg_csv and Path(deg_csv).exists():
        payload["deg_list"] = load_deg_list_from_csv(deg_csv)
        print(f"Loaded {len(payload['deg_list'])} DEGs from {deg_csv}")
    
    # Load pathways
    if pathways_csv and Path(pathways_csv).exists():
        payload["pathways"] = load_pathways_from_csv(pathways_csv)
        print(f"Loaded {len(payload['pathways'])} pathways from {pathways_csv}")
    
    # Generate phenotypes if not provided and we have DEGs/pathways
    if not phenotypes and payload["deg_list"] and payload["pathways"]:
        try:
            from llm import PhenotypeKB
            kb = PhenotypeKB()  # Empty KB - will score based on DEGs/pathways
            payload["phenotypes"] = kb.score_phenotypes(
                payload["deg_list"],
                payload["pathways"]
            )
            print(f"Generated {len(payload['phenotypes'])} phenotypes from DEGs and pathways")
        except ImportError:
            # Try relative import
            from phenotype_kb import PhenotypeKB
            kb = PhenotypeKB()
            payload["phenotypes"] = kb.score_phenotypes(
                payload["deg_list"],
                payload["pathways"]
            )
            print(f"Generated {len(payload['phenotypes'])} phenotypes from DEGs and pathways")
    
    return payload


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load data from CSV and create payload")
    parser.add_argument("--deg-csv", type=str, help="Path to DEG CSV file")
    parser.add_argument("--pathways-csv", type=str, help="Path to pathways CSV file")
    parser.add_argument("--output", type=str, default="payload.json", help="Output JSON file")
    parser.add_argument("--perturbation", type=str, default="Unknown", help="Perturbation name")
    parser.add_argument("--cell-type", type=str, default="", help="Cell type")
    parser.add_argument("--user-question", type=str, default="", help="User question")
    
    args = parser.parse_args()
    
    # Create context
    context = {
        "perturbation": args.perturbation,
        "cell_type": args.cell_type,
        "species": "human",
        "user_question": args.user_question
    }
    
    # Create payload
    payload = create_payload_from_files(
        deg_csv=args.deg_csv,
        pathways_csv=args.pathways_csv,
        context=context
    )
    
    # Save payload
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"\nPayload saved to: {args.output}")
    print(f"  - {len(payload['deg_list'])} DEGs")
    print(f"  - {len(payload['pathways'])} pathways")
    print(f"  - {len(payload['phenotypes'])} phenotypes")


if __name__ == "__main__":
    main()

