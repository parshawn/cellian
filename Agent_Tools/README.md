# Agent Tools: Perturbation → Transcriptomics → Proteomics Pipeline

This directory contains tools for running a three-node graph pipeline:
1. **Perturbation** → **Transcriptomics (RNA)** using the state model
2. **Transcriptomics (RNA)** → **Proteomics** using scTranslator

## Overview

The pipeline takes a perturbation (gene name) as input and predicts:
- RNA expression from perturbation (using state model)
- Protein expression from RNA (using scTranslator)

At each step, predictions are evaluated against ground truth data using R2 scores and other metrics.

## Files

- `perturbation_pipeline.py`: Main CLI tool that orchestrates the entire pipeline
- `state_inference.py`: Module for running state inference (gene perturbation → RNA)
- `drug_inference.py`: Module for running ST-Tahoe inference (drug perturbation → RNA)
- `sctranslator_inference.py`: Module for running scTranslator inference (RNA → protein)
- `sctranslator_inference_custom.py`: Custom scTranslator inference that handles fine-tuned checkpoints
- `evaluation.py`: Evaluation metrics (R2, Pearson correlation, RMSE, MAE)
- `pathway_analysis.py`: Pathway and enrichment analysis (GSEA, KEGG, Reactome, GO)
- `requirements_analysis.txt`: Additional requirements for pathway analysis

## Usage

### Interactive Mode (Recommended)

Simply run the pipeline without arguments, and it will prompt you to choose:

```bash
cd /home/nebius/cellian/Agent_Tools
python perturbation_pipeline.py
```

The pipeline will ask:
1. Choose perturbation type (Gene or Drug)
2. Enter the perturbation name

### Command-Line Mode

**Gene Perturbation:**
```bash
python perturbation_pipeline.py --target-gene ACTB
python perturbation_pipeline.py --target-gene AARS
```

The pipeline will show you available gene perturbations from the STATE model when you select gene perturbation (similar to drugs).

**Drug Perturbation:**
```bash
python perturbation_pipeline.py --perturbation-type drug --drug DRUG_NAME
```

**Note:** Drug perturbations must be input in tuple format: `"[('drugname', concentration, 'uM')]"` 
(e.g., `"[('(R)-Verapamil (hydrochloride)', 0.05, 'uM')]"`)


The pipeline:
- Filters to control cells (non-targeting) from perturb-cite-seq data before applying drug perturbation (biologically meaningful)
- Creates a `drugname_drugconc` column with:
  - 80% of cells set to the user's desired drug perturbation
  - 20% of cells set to control: `"[('DMSO_TF', 0.0, 'uM')]"`
- This allows for GSEA and pathway enrichment analysis by comparing perturbed vs control cells
- Does NOT perform validation/evaluation against perturb-cite-seq data (no ground truth comparison)
- Runs inference with checkpoint and feeds results to scTranslator
- Performs comprehensive pathway analysis on the resulting predictions:
  - Correlation analysis between RNA and protein
  - Differential expression analysis (RNA and protein)
  - GSEA (Gene Set Enrichment Analysis) on RNA
  - Pathway enrichment analysis (KEGG, Reactome, GO) on protein
- Uses ST-Tahoe model with checkpoint parameter

### With Custom Paths

**Gene Perturbation:**
```bash
python perturbation_pipeline.py --target-gene ACTB \
  --state-model-dir /path/to/state/model \
  --state-checkpoint /path/to/state/checkpoint.ckpt \
  --sctranslator-checkpoint /path/to/sctranslator.pt \
  --control-template /path/to/control_template.h5ad
```

**Drug Perturbation:**
```bash
python perturbation_pipeline.py --perturbation-type drug --drug DRUG_NAME \
  --st-tahoe-model-dir /path/to/ST-Tahoe \
  --drug-data /path/to/drug_data.h5ad \
  --sctranslator-checkpoint /path/to/sctranslator.pt
```

### Options

**Perturbation Type:**
- `--perturbation-type`: Type of perturbation: `gene` (default) or `drug`

**Gene Perturbation Options:**
- `--target-gene` or `--perturbation`: Target gene name (required for gene perturbation)
- `--state-model-dir`: Path to state model directory (default: `/home/nebius/state/test_replogle/hepg2_holdout`)
- `--state-checkpoint`: Path to state checkpoint (default: `.../checkpoints/last.ckpt`)
- `--control-template`: Path to control template h5ad (default: `.../scp1064_control_template.h5ad`)

**Drug Perturbation Options:**
- `--drug`: Drug name in format `drugname_drugconc` (required for drug perturbation, e.g., `aspirin_10uM`)
- `--st-tahoe-model-dir`: Path to ST-Tahoe model directory (default: `/home/nebius/ST-Tahoe`)
- `--drug-data`: Path to drug data file (default: `.../scp1064_aligned_to_model_tahoe.h5ad`)

**Common Options:**
- `--sctranslator-checkpoint`: Path to scTranslator checkpoint (default: `/home/nebius/scTranslator/checkpoint/expression_fine-tuned_scTranslator.pt`)
- `--ground-truth-rna`: Path to ground truth RNA (default: `.../RNA_expression_combined_mapped.h5ad`)
- `--ground-truth-protein`: Path to ground truth protein for evaluation (default: `.../protein_expression_mapped.h5ad`)
- `--protein-file`: Path to protein expression file for scTranslator (default: `.../protein_expression.h5ad`)
- `--output-dir`: Output directory (default: `Agent_Tools/temp_output`)
- `--skip-evaluation`: Skip evaluation against ground truth
- `--keep-temp`: Keep temporary files after completion

## Pipeline Steps

1. **Prepare Perturbation Data**: 
   - For gene perturbation: Creates perturbation data from control template
   - For drug perturbation: Filters drug data to cells with specified drug
2. **Inference (Perturbation → RNA)**: 
   - For gene perturbation: Runs state model to predict RNA
   - For drug perturbation: Runs ST-Tahoe model to predict RNA
3. **Evaluate RNA Predictions**: Compares predicted RNA to ground truth (only cells with the perturbation)
4. **Prepare RNA for scTranslator**: Maps gene IDs to scTranslator vocabulary
5. **Prepare Protein Data**: Filters real protein file to match RNA predictions (same perturbation) and aligns cells
6. **scTranslator Inference**: Runs scTranslator to predict protein from RNA
7. **Evaluate Protein Predictions**: Compares predicted protein to ground truth (only cells with the perturbation)
8. **Pathway and Enrichment Analysis**: Comprehensive analysis including:
   - Correlation analysis between RNA and protein
   - Differential expression analysis (RNA vs control, protein vs control)
   - Gene Set Enrichment Analysis (GSEA) on RNA
   - Pathway enrichment analysis (KEGG, Reactome, GO) on protein
9. **Save Results**: Saves all predictions, evaluations, and pathway analyses

## Output

Results are saved to the output directory (default: `Agent_Tools/temp_output`):
- `perturbation_{GENE}.h5ad`: Perturbation data
- `rna_predictions_{GENE}.h5ad`: Predicted RNA expression
- `rna_predictions_{GENE}_mapped.h5ad`: RNA with scTranslator gene IDs
- `protein_prepared_{GENE}.h5ad`: Prepared protein data (filtered and aligned with RNA)
- `protein_prepared_{GENE}_mapped.h5ad`: Prepared protein with scTranslator gene IDs
- `protein_predictions_{GENE}.h5ad`: Predicted protein expression
- `pathway_analysis/`: Pathway analysis results:
  - `correlation_analysis_{GENE}.csv`: Correlation between RNA and protein
  - `differential_expression_rna_{GENE}.csv`: Differential expression (RNA vs control)
  - `differential_expression_protein_{GENE}.csv`: Differential expression (protein vs control)
  - `gsea_rna_{GENE}.csv`: GSEA results for RNA
  - `kegg_enrichment_{GENE}.csv`: KEGG pathway enrichment for protein
  - `reactome_enrichment_{GENE}.csv`: Reactome pathway enrichment for protein
  - `go_enrichment_{GENE}.csv`: GO term enrichment for protein

## Evaluation Metrics

- **R2 Score**: Coefficient of determination
- **Pearson Correlation**: Linear correlation coefficient
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error

## Requirements

### Core Requirements
- Python 3.8+
- anndata
- pandas
- numpy
- scanpy
- torch
- sklearn
- scipy
- state (command-line tool)
- scTranslator (with fine-tuned checkpoint)

### Pathway Analysis Requirements (Optional)
For pathway and enrichment analysis, install additional packages:
```bash
pip install -r requirements_analysis.txt
```

Or install individually:
```bash
pip install gseapy statsmodels matplotlib seaborn networkx goatools
```

Note: Pathway analysis will be skipped if required packages are not installed, but the pipeline will continue.

## Pathway Analysis

The pipeline includes comprehensive pathway and enrichment analysis:

### 1. Correlation Analysis
- Calculates correlation between RNA and protein predictions
- Identifies genes/proteins with strong correlations
- Provides per-gene correlation statistics

### 2. Differential Expression Analysis
- Compares predicted RNA vs control RNA
- Compares predicted protein vs control protein
- Calculates log2 fold change, p-values, and adjusted p-values
- Identifies significantly up/downregulated genes/proteins

### 3. Gene Set Enrichment Analysis (GSEA)
- Runs GSEA on RNA differential expression
- Uses KEGG pathways by default
- Identifies enriched pathways from perturbation

### 4. Pathway Enrichment Analysis
- **KEGG**: Kyoto Encyclopedia of Genes and Genomes pathways
- **Reactome**: Reactome pathway database
- **GO**: Gene Ontology biological processes
- All analyses run on protein differential expression

### Running Pathway Analysis Separately

If you want to run pathway analysis separately:

```bash
python pathway_analysis.py \
  --rna-predictions /path/to/rna_predictions.h5ad \
  --protein-predictions /path/to/protein_predictions.h5ad \
  --control-rna /path/to/control_rna.h5ad \
  --control-protein /path/to/control_protein.h5ad \
  --target-gene ACTB \
  --output-dir /path/to/output
```

## Notes

- The state model uses 2000 highly variable genes (HVGs)
- scTranslator requires gene ID mapping (my_Id column)
- Ground truth data is used for evaluation but is optional
- The pipeline handles checkpoint format differences between fine-tuned and pre-trained models
- Pathway analysis requires additional packages (see Requirements section)
- Pathway analysis is automatically run if ground truth data is provided and evaluation is enabled
- Protein file is filtered by target_gene and aligned with RNA predictions automatically
- If cells don't match between RNA and protein, protein structure is used with RNA cell IDs

