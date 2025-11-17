"""Test script for LLM workflow with fake/sample data."""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import (
    generate_hypotheses,
    plot_pathway_enrichment,
    plot_phenotype_scores,
    plot_volcano,
    plot_rna_gsea,
    plot_protein_psea,
    plot_phenotype_enrichment,
    build_report,
    PhenotypeKB
)


def create_sample_payload() -> dict:
    """Create sample payload matching the expected schema."""
    return {
        "context": {
            "perturbation": "KO TP53",
            "cell_type": "HepG2",
            "species": "human",
            "user_question": "What happens if I knock out TP53 in HepG2 cells?"
        },
        "validated_edges": [
            {"source": "TP53", "target": "CDKN1A", "direction": "down", "confidence": 0.85},
            {"source": "TP53", "target": "BAX", "direction": "down", "confidence": 0.82},
            {"source": "TP53", "target": "MDM2", "direction": "down", "confidence": 0.78},
            {"source": "TP53", "target": "P21", "direction": "down", "confidence": 0.90},
            {"source": "CDKN1A", "target": "CCND1", "direction": "up", "confidence": 0.75},
            {"source": "BAX", "target": "CASP3", "direction": "down", "confidence": 0.70}
        ],
        "deg_list": [
            {"gene": "CDKN1A", "log2fc": -2.5, "pval": 0.001},
            {"gene": "BAX", "log2fc": -2.1, "pval": 0.003},
            {"gene": "MDM2", "log2fc": -1.8, "pval": 0.005},
            {"gene": "P21", "log2fc": -2.3, "pval": 0.002},
            {"gene": "CCND1", "log2fc": 1.5, "pval": 0.01},
            {"gene": "CASP3", "log2fc": -1.2, "pval": 0.02},
            {"gene": "BCL2", "log2fc": 1.3, "pval": 0.015},
            {"gene": "PCNA", "log2fc": 1.8, "pval": 0.008},
            {"gene": "GADD45A", "log2fc": -1.5, "pval": 0.012},
            {"gene": "FAS", "log2fc": -0.9, "pval": 0.025},
            {"gene": "MYC", "log2fc": 1.2, "pval": 0.018},
            {"gene": "RB1", "log2fc": -1.1, "pval": 0.022},
            {"gene": "ATM", "log2fc": -0.8, "pval": 0.03},
            {"gene": "CHEK2", "log2fc": -1.0, "pval": 0.028},
            {"gene": "PTEN", "log2fc": -0.7, "pval": 0.035}
        ],
        "pathways": [
            {
                "id": "KEGG_P53_PATHWAY",
                "name": "p53 signaling pathway",
                "source": "GSEA",
                "NES": 2.5,
                "FDR": 0.001,
                "member_genes": ["TP53", "CDKN1A", "BAX", "MDM2", "P21", "ATM", "CHEK2"]
            },
            {
                "id": "KEGG_CELL_CYCLE",
                "name": "Cell cycle",
                "source": "GSEA",
                "NES": 2.1,
                "FDR": 0.003,
                "member_genes": ["CDKN1A", "CCND1", "RB1", "PCNA", "MYC"]
            },
            {
                "id": "KEGG_APOPTOSIS",
                "name": "Apoptosis",
                "source": "GSEA",
                "NES": -2.3,
                "FDR": 0.002,
                "member_genes": ["BAX", "BCL2", "CASP3", "FAS"]
            },
            {
                "id": "KEGG_DNA_REPAIR",
                "name": "DNA repair",
                "source": "PSEA",
                "NES": -1.8,
                "FDR": 0.01,
                "member_genes": ["ATM", "CHEK2", "GADD45A", "TP53"]
            },
            {
                "id": "REACTOME_CELL_CYCLE_CHECKPOINTS",
                "name": "Cell cycle checkpoints",
                "source": "GSEA",
                "NES": 1.9,
                "FDR": 0.008,
                "member_genes": ["TP53", "CDKN1A", "RB1", "ATM"]
            }
        ],
        "phenotypes": [
            {
                "phenotype_id": "HP:0001903",
                "name": "Increased apoptosis",
                "score": 0.75,
                "direction": "increase",
                "supporting_genes": ["BAX", "CASP3", "FAS"],
                "supporting_up_genes": [],
                "supporting_down_genes": ["BAX", "CASP3", "FAS"],
                "supporting_pathways": [
                    {"id": "KEGG_APOPTOSIS", "name": "Apoptosis", "NES": -2.3, "FDR": 0.002}
                ]
            },
            {
                "phenotype_id": "HP:0000086",
                "name": "Increased cell proliferation",
                "score": 0.82,
                "direction": "increase",
                "supporting_genes": ["CCND1", "PCNA", "MYC"],
                "supporting_up_genes": ["CCND1", "PCNA", "MYC"],
                "supporting_down_genes": [],
                "supporting_pathways": [
                    {"id": "KEGG_CELL_CYCLE", "name": "Cell cycle", "NES": 2.1, "FDR": 0.003}
                ]
            },
            {
                "phenotype_id": "HP:0002814",
                "name": "Abnormal DNA repair",
                "score": 0.68,
                "direction": "decrease",
                "supporting_genes": ["ATM", "CHEK2", "GADD45A", "TP53"],
                "supporting_up_genes": [],
                "supporting_down_genes": ["ATM", "CHEK2", "GADD45A"],
                "supporting_pathways": [
                    {"id": "KEGG_DNA_REPAIR", "name": "DNA repair", "NES": -1.8, "FDR": 0.01}
                ]
            },
            {
                "phenotype_id": "HP:0002664",
                "name": "Neoplasia",
                "score": 0.71,
                "direction": "increase",
                "supporting_genes": ["TP53", "MYC", "CCND1", "RB1"],
                "supporting_up_genes": ["MYC", "CCND1"],
                "supporting_down_genes": ["TP53", "RB1"],
                "supporting_pathways": [
                    {"id": "KEGG_P53_PATHWAY", "name": "p53 signaling pathway", "NES": 2.5, "FDR": 0.001},
                    {"id": "KEGG_CELL_CYCLE", "name": "Cell cycle", "NES": 2.1, "FDR": 0.003}
                ]
            },
            {
                "phenotype_id": "HP:0000952",
                "name": "Cell cycle arrest",
                "score": 0.58,
                "direction": "decrease",
                "supporting_genes": ["CDKN1A", "P21", "RB1"],
                "supporting_up_genes": [],
                "supporting_down_genes": ["CDKN1A", "P21", "RB1"],
                "supporting_pathways": [
                    {"id": "KEGG_P53_PATHWAY", "name": "p53 signaling pathway", "NES": 2.5, "FDR": 0.001},
                    {"id": "REACTOME_CELL_CYCLE_CHECKPOINTS", "name": "Cell cycle checkpoints", "NES": 1.9, "FDR": 0.008}
                ]
            }
        ],
        "evidence": {
            "datasets": ["SCP1064 perturbation-CITE-seq"],
            "papers": []
        }
    }


def test_workflow():
    """Test the complete LLM workflow."""
    print("=" * 70)
    print("TESTING LLM WORKFLOW WITH SAMPLE DATA")
    print("=" * 70)
    
    # Create sample payload
    print("\n1. Creating sample payload...")
    payload = create_sample_payload()
    print(f"   ✓ Created payload with:")
    print(f"     - {len(payload['deg_list'])} DEGs")
    print(f"     - {len(payload['pathways'])} pathways")
    print(f"     - {len(payload['phenotypes'])} phenotypes")
    print(f"     - {len(payload['validated_edges'])} validated edges")
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    print(f"\n   Output directory: {output_dir}")
    
    # Save sample payload as JSON
    json_path = output_dir / "sample_payload.json"
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"   ✓ Saved sample payload to: {json_path}")
    
    # Test 1: Generate plots
    print("\n2. Generating plots...")
    plot_paths = {}
    
    try:
        volcano_path = output_dir / "volcano.png"
        plot_paths["volcano"] = str(plot_volcano(payload["deg_list"], str(volcano_path)))
        print(f"   ✓ Volcano plot: {plot_paths['volcano']}")
    except Exception as e:
        print(f"   ✗ Volcano plot failed: {e}")
        plot_paths["volcano"] = None
    
    try:
        pathway_path = output_dir / "pathway_enrichment.png"
        plot_paths["pathway_enrichment"] = str(plot_pathway_enrichment(
            payload["pathways"],
            str(pathway_path)
        ))
        print(f"   ✓ Pathway enrichment plot: {plot_paths['pathway_enrichment']}")
    except Exception as e:
        print(f"   ✗ Pathway plot failed: {e}")
        plot_paths["pathway_enrichment"] = None
    
    try:
        phenotype_path = output_dir / "phenotype_scores.png"
        plot_paths["phenotype_scores"] = str(plot_phenotype_scores(
            payload["phenotypes"],
            str(phenotype_path)
        ))
        print(f"   ✓ Phenotype scores plot: {plot_paths['phenotype_scores']}")
    except Exception as e:
        print(f"   ✗ Phenotype plot failed: {e}")
        plot_paths["phenotype_scores"] = None
    
    # RNA GSEA - Real gene-set enrichment on transcriptomics
    try:
        rna_gsea_path = output_dir / "rna_gsea.png"
        plot_paths["rna_gsea"] = str(plot_rna_gsea(
            payload["deg_list"],  # RNA DEGs from transcriptomics
            payload["pathways"],
            str(rna_gsea_path),
            top_pathways=20
        ))
        print(f"   ✓ RNA GSEA plot: {plot_paths['rna_gsea']}")
    except Exception as e:
        print(f"   ✗ RNA GSEA plot failed: {e}")
        import traceback
        traceback.print_exc()
        plot_paths["rna_gsea"] = None
    
    # Protein/PPI PSEA - Network-aware enrichment on proteomics/PPI
    try:
        protein_psea_path = output_dir / "protein_psea.png"
        plot_paths["protein_psea"] = str(plot_protein_psea(
            payload["deg_list"],  # Using same DEGs as proxy for protein (in real use, use protein_deg_list)
            payload["validated_edges"],  # PPI network edges
            payload["pathways"],
            str(protein_psea_path),
            top_pathways=20
        ))
        print(f"   ✓ Protein/PPI PSEA plot: {plot_paths['protein_psea']}")
    except Exception as e:
        print(f"   ✗ Protein/PPI PSEA plot failed: {e}")
        import traceback
        traceback.print_exc()
        plot_paths["protein_psea"] = None
    
    # Phenotype Enrichment
    try:
        phenotype_enrichment_path = output_dir / "phenotype_enrichment.png"
        # Initialize PhenotypeKB (empty for testing)
        phenotype_kb = PhenotypeKB()
        plot_paths["phenotype_enrichment"] = str(plot_phenotype_enrichment(
            payload["deg_list"],
            payload["phenotypes"],
            phenotype_kb=phenotype_kb,
            out_path=str(phenotype_enrichment_path),
            top_phenotypes=5
        ))
        print(f"   ✓ Phenotype enrichment plot: {plot_paths['phenotype_enrichment']}")
    except Exception as e:
        print(f"   ✗ Phenotype enrichment plot failed: {e}")
        import traceback
        traceback.print_exc()
        plot_paths["phenotype_enrichment"] = None
    
    # Test 2: Generate hypotheses
    print("\n3. Generating hypotheses...")
    try:
        hypotheses = generate_hypotheses(payload)
        print(f"   ✓ Generated {len(hypotheses.get('hypotheses', []))} hypotheses")
        
        # Save hypotheses
        hypotheses_path = output_dir / "hypotheses.json"
        with open(hypotheses_path, 'w') as f:
            json.dump(hypotheses, f, indent=2)
        print(f"   ✓ Saved hypotheses to: {hypotheses_path}")
        
        # Print hypothesis summaries
        for i, hyp in enumerate(hypotheses.get("hypotheses", []), 1):
            print(f"\n   Hypothesis {hyp.get('id', f'H{i}')}:")
            print(f"     Statement: {hyp.get('statement', 'N/A')}")
            print(f"     Literature support: {hyp.get('literature_support', {}).get('overall', 'unknown')}")
            
    except Exception as e:
        print(f"   ✗ Hypothesis generation failed: {e}")
        import traceback
        traceback.print_exc()
        hypotheses = {"hypotheses": []}
    
    # Test 3: Generate report
    print("\n4. Generating report...")
    try:
        report = build_report(
            context=payload["context"],
            pathways=payload["pathways"],
            phenotypes=payload["phenotypes"],
            hypotheses=hypotheses,
            plot_paths=plot_paths
        )
        
        # Save report
        report_path = output_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"   ✓ Generated report: {report_path}")
        print(f"   Report length: {len(report)} characters")
        
    except Exception as e:
        print(f"   ✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Phenotype KB (with empty data - just test the structure)
    print("\n5. Testing PhenotypeKB structure...")
    try:
        kb = PhenotypeKB()  # Empty KB for testing
        gene_phenos = kb.get_gene_phenotypes("TP53")
        print(f"   ✓ PhenotypeKB initialized (empty data)")
        print(f"   ✓ get_gene_phenotypes('TP53') returned {len(gene_phenos)} phenotypes")
        
        # Test scoring with sample data
        scored = kb.score_phenotypes(
            payload["deg_list"],
            payload["pathways"]
        )
        print(f"   ✓ score_phenotypes() returned {len(scored)} scored phenotypes")
        
    except Exception as e:
        print(f"   ✗ PhenotypeKB test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    print(f"  - {json_path.name}")
    if plot_paths.get("volcano"):
        print(f"  - volcano.png")
    if plot_paths.get("pathway_enrichment"):
        print(f"  - pathway_enrichment.png")
    if plot_paths.get("phenotype_scores"):
        print(f"  - phenotype_scores.png")
    if plot_paths.get("rna_gsea"):
        print(f"  - rna_gsea.png")
    if plot_paths.get("protein_psea"):
        print(f"  - protein_psea.png")
    if plot_paths.get("phenotype_enrichment"):
        print(f"  - phenotype_enrichment.png")
    if os.path.exists(output_dir / "hypotheses.json"):
        print(f"  - hypotheses.json")
    if os.path.exists(output_dir / "report.md"):
        print(f"  - report.md")
    
    print("\n✓ Workflow test completed!")
    print(f"\nView results in: {output_dir}")


if __name__ == "__main__":
    test_workflow()

