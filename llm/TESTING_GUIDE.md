# Testing Guide for Perturbation Orchestrator

This guide explains how to test the perturbation orchestrator workflow, including the new intelligent query features.

## Test Scripts Available

1. **`test_orchestrator.py`** - Comprehensive test suite with MOCKED Agent_Tools (fast, no real pipeline execution)
2. **`test_real_workflow.py`** - Tests REAL workflow with actual Agent_Tools (takes 20-40 min per perturbation)
3. **`test_intelligent_queries.py`** - Focused tests for intelligent query features (protein focus, top N, etc.)
4. **`test_workflow.py`** - Tests LLM workflow with sample data (plot generation, hypothesis generation)
5. **`run_orchestrator_slurm.sh`** - SLURM script for real workflow execution (RECOMMENDED for real testing)

## Quick Test (Fast - Mocked Agent_Tools)

**Use this for:** Testing orchestrator logic, intent detection, and output generation WITHOUT running actual Agent_Tools pipelines.

### Run locally:
```bash
cd /home/nebius/cellian
python llm/test_orchestrator.py
```

### Test intelligent queries specifically:
```bash
python llm/test_intelligent_queries.py
```

## Real Workflow Test (With Actual Agent_Tools)

**Use this for:** Testing the COMPLETE workflow with REAL Agent_Tools execution (takes 20-40 minutes per perturbation).

### Option 1: Test locally (if you have Agent_Tools set up):
```bash
cd /home/nebius/cellian

# Test integration check only:
python llm/test_real_workflow.py

# Test with a real query (will run actual Agent_Tools):
python llm/test_real_workflow.py "run KO TP53"

# Test with a protein-focused query:
python llm/test_real_workflow.py "Show me top 10 proteins changed by TP53 knockout" --condition Control
```

### Option 2: Test on SLURM (RECOMMENDED for real Agent_Tools execution):

#### 1. Test with a simple query:
```bash
sbatch llm/run_orchestrator_slurm.sh "run KO TP53"
```

#### 2. Test with protein-focused query:
```bash
sbatch llm/run_orchestrator_slurm.sh "Show me top 10 proteins changed by TP53 knockout"
```

#### 3. Test with top N filtering:
```bash
sbatch llm/run_orchestrator_slurm.sh "What are the top 5 genes and top 3 pathways affected?"
```

#### 4. Test comparison with protein focus:
```bash
sbatch llm/run_orchestrator_slurm.sh "Compare protein changes between TP53 KO and imatinib"
```

#### 5. Test pathway-based batch query:
```bash
sbatch llm/run_orchestrator_slurm.sh "compare the effect of top 5 relevant drugs and gene knockdowns affecting the mTOR pathway that affects cell proliferation"
```

## Example Queries to Test

### Protein-focused queries:
- "Show me top 10 proteins changed by TP53 knockout"
- "What proteins change when I knock down JAK1?"
- "Show me the top 7 most changed proteins and their pathways"
- "Compare protein changes between TP53 KO and imatinib"

### Top N filtering queries:
- "What are the top 5 genes affected?"
- "Show me top 10 proteins changed by TP53 knockout"
- "Find top 3 pathways affected"
- "What are the top 5 genes and top 3 pathways affected?"

### Pathway-based queries:
- "Find top 3 genes in PI3K pathway affecting apoptosis"
- "Top 10 genes in mTOR pathway"
- "compare the effect of top 5 relevant drugs and gene knockdowns affecting the mTOR pathway that affects cell proliferation"

### Comparison queries:
- "Compare TP53 knockout versus imatinib"
- "Which perturbation is stronger: KO of TP53 or imatinib?"
- "Compare protein changes between TP53 KO and imatinib"

## Check Results

After running, check the output:
```bash
# View logs
ls -lh llm/logs/

# View latest log
tail -f llm/logs/orchestrator_*.out

# Check results directory
ls -lh llm/perturbation_outputs/

# View results for a specific perturbation
ls -lh llm/perturbation_outputs/gene_TP53/
ls -lh llm/perturbation_outputs/drug_*/
```

## Test Output Structure

After running a query, you should see:
```
llm/perturbation_outputs/
├── gene_TP53/          (for gene perturbations)
│   ├── volcano.png
│   ├── pathway_enrichment.png
│   ├── rna_gsea.png
│   ├── protein_psea.png
│   ├── phenotype_enrichment.png
│   ├── phenotype_scores.png
│   ├── hypotheses.json
│   ├── report.md
│   ├── top_genes.csv      (if top N requested)
│   ├── top_proteins.csv   (if top N proteins requested)
│   └── top_pathways.csv   (if top N pathways requested)
├── drug_*/              (for drug perturbations)
│   └── ...
└── comparison/           (for comparison queries)
    ├── comparison_report.md
    └── comparison.json
```

## What the Tests Verify

### `test_orchestrator.py`:
- ✅ Name validation (exact/close/none matching)
- ✅ User intent detection
- ✅ Mock pipeline execution
- ✅ Output generation
- ✅ Comparison logic
- ✅ Intelligent query extraction (NEW)

### `test_intelligent_queries.py`:
- ✅ Protein-focused query detection
- ✅ Top N extraction (genes, proteins, pathways, phenotypes)
- ✅ Adaptive output generation based on query intent
- ✅ Focus detection (genes, proteins, pathways, phenotypes, both)
- ✅ Output type detection

### `test_workflow.py`:
- ✅ Plot generation with sample data
- ✅ Hypothesis generation
- ✅ Report generation
- ✅ PhenotypeKB functionality

## Troubleshooting

### If tests fail:
1. Check that all dependencies are installed (pandas, matplotlib, etc.)
2. Check that pickle files exist for perturbation names
3. Check logs in `llm/logs/` for detailed error messages
4. Verify environment variables (GEMINI_API_KEY for LLM features)

### If SLURM jobs fail:
1. Check job output: `cat llm/logs/orchestrator_*.out`
2. Check job errors: `cat llm/logs/orchestrator_*.err`
3. Verify conda environment is activated
4. Check that Agent_Tools paths are correct

## Next Steps

After testing, you can:
1. Run real queries on SLURM with actual Agent_Tools
2. Customize queries for your specific use cases
3. Add more test cases for edge cases
4. Extend the intelligent query extraction for domain-specific terms
