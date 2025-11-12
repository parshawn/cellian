# Mechanistic Bio Reasoning Engine

A Python 3.10+ project that uses an LLM to plan tool calls and Python to execute them. The system implements a 3-node causal reasoning chain (Perturbation → RNA → Protein) with autonomous hypothesis generation.

## Features

- **HypothesisAgent**: Autonomous agent that traverses the causal graph and generates hypotheses
- **3-Node Chain**: Pre-computed aligned perturbation embedding → RNA prediction → Protein prediction
- **Graph Structure**: Causal graph with nodes and edges representing the reasoning chain
- **Pre-aligned Embeddings**: Assumes embeddings are already computed and aligned (no alignment work)
- **Ready Nodes**: Works with pre-computed nodes and edges (graph traversal only)
- **Validation**: Compares predictions to ground truth and calculates error propagation
- **Data Integration**: Loads real RNA/protein profiles from perturbation-CITE-seq data

## Setup

1. Install dependencies:
```bash
pip install google-generativeai pandas numpy torch python-dotenv
```

2. Set your Gemini API key (for LLM planner):
   
   **Option 1: Using .env file (Recommended)**
   ```bash
   # Copy the example file and add your API key
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```
   
   **Option 2: Using environment variable**
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```
   
   The code will automatically load the API key from the `.env` file if `python-dotenv` is installed (included in requirements.txt).
   
   **Note**: Install the Google Generative AI package:
   ```bash
   pip install google-generativeai
   ```

3. Ensure data files are available:
   - `/home/nebius/cellian/data/perturb-cite-seq/SCP1064/` (RNA/protein expression data)
   - `/home/nebius/cellian/outputs/cell_embeddings.pt` (required: pre-computed aligned embeddings)
   
   **Note**: Embeddings must be pre-computed and aligned before using the reasoning_layer.
   The reasoning_layer assumes embeddings are already aligned and nodes are ready.

## Run

### LLM Agent (Recommended - Natural Language Queries)

From the `reasoning_layer` directory:

```bash
python run_llm_agent.py "What happens if I knock down JAK1?"
```

Examples:
```bash
# Natural language query
python run_llm_agent.py "What happens if I knock out STAT1?"

# With dummy graph
python run_llm_agent.py "What happens if I knock down JAK1?" --dummy
```

**Features:**
- Takes natural language questions
- Extracts perturbation information automatically
- Executes the 3-node chain
- Generates LLM interpretation of results
- Returns comprehensive analysis

### HypothesisAgent (Direct Usage)

```bash
python run_hypothesis_agent.py <perturbation_name>
```

Example:
```bash
python run_hypothesis_agent.py JAK1
```

**Using dummy graph for testing:**
```bash
python run_hypothesis_agent.py JAK1 --dummy
```

**Create and save a dummy graph:**
```bash
python create_dummy_graph.py JAK1
```

### Basic Demo (LLM Planner)

**Normal mode (requires real data):**
```bash
python run_demo.py
```

**Test mode (arbitrary data, no DataLoader required):**
```bash
# Basic test with default arbitrary data
python run_demo.py --test

# Test with custom parameters
python run_demo.py --test --perturbation JAK1_KO --genes "STAT1,IRF1,HLA-A" --proteins "HLA-A,CD58"

# Test with perfect match (for testing metrics)
python run_demo.py --test --perfect-match

# Test with custom seed for reproducibility
python run_demo.py --test --seed 42

# See full options
python run_demo.py --test --help
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing instructions.

## Architecture

### 3-Node Chain

1. **Node 1: Pre-computed Aligned Perturbation Embedding**
   - Loads pre-computed aligned perturbation embedding
   - Tool: `perturbation.get_embedding`
   - Assumes embedding is already aligned and ready

2. **Node 2: RNA Prediction**
   - Predicts RNA changes from pre-computed aligned embedding
   - Tool: `state.predict`
   - Uses STATE model with aligned embedding (or mock)

3. **Node 3: Protein Prediction**
   - Predicts protein changes from RNA prediction
   - Tool: `captain.translate`
   - Uses CAPTAIN model (or mock)

### Graph Structure

The `CausalGraph` class maintains:
- **Nodes**: Represent perturbation, RNA, and protein predictions
- **Edges**: Represent transformations (embedding → RNA → Protein)
- **Metadata**: Embeddings, predictions, and validation scores

**Graph Loading:**
- **Pre-computed graphs**: Load from JSON files (`/home/nebius/cellian/outputs/graphs/`)
- **Dummy graphs**: Create dummy graphs for testing
- **Dynamic graphs**: Build graphs on-the-fly during execution

**Graph Modes:**
1. **Pre-computed**: Load existing graph from file
2. **Dummy**: Create dummy graph with test data
3. **Dynamic**: Build graph as agent executes (default)

### HypothesisAgent

The `HypothesisAgent` class:
- **Assumes pre-computed aligned embeddings and ready nodes**
- Fetches control cell baselines
- Loads pre-computed aligned embeddings (no alignment work)
- Executes the 3-node chain using aligned embeddings
- Traverses causal graph with pre-computed nodes and edges
- Compares predictions to ground truth
- Calculates error propagation
- Returns structured JSON with hypothesis and validation

**Important**: The reasoning_layer does NOT compute or align embeddings.
It assumes embeddings are already computed and aligned elsewhere.

## Output Format

### LLM Agent Output

The LLMAgent returns a JSON object with:
- `query`: Original natural language query
- `perturbation_info`: Extracted perturbation information (target, type, confidence)
- `llm_interpretation`: **LLM-generated natural language summary and interpretation**
- `hypothesis`: Text description of the hypothesis
- `path`: 3-node path with execution details
- `graph`: Causal graph structure (nodes and edges)
- `predictions`: Predicted RNA and protein deltas
- `ground_truth`: Real RNA and protein deltas
- `validation_scores`: 
  - RNA: Spearman correlation, MSE, MAE
  - Protein: Pearson correlation, MSE, MAE
- `error_propagation`: Error propagation through the chain
- `metadata`: Execution metadata

### HypothesisAgent Output

Same as above, but without `query`, `perturbation_info`, and `llm_interpretation`.

See [METRICS.md](METRICS.md) for detailed information about the metrics.

## Files

- `engine/llm_agent.py`: **LLM-powered agent for natural language queries**
- `engine/llm_interpreter.py`: **LLM interpreter for results summarization**
- `engine/planner.py`: LLM planner for tool execution plans
- `engine/hypothesis_agent.py`: Main HypothesisAgent class
- `engine/causal_graph.py`: Graph structure for nodes and edges
- `engine/graph_loader.py`: Graph loading and dummy graph creation
- `engine/data_loader.py`: Loads control cells and real profiles
- `engine/tools_perturbation.py`: Perturbation embedding tool (Node 1)
- `engine/tools_state.py`: RNA prediction tool (Node 2)
- `engine/tools_captain.py`: Protein prediction tool (Node 3)
- `engine/profile_comparison.py`: Profile comparison and error propagation
- `engine/embedding_utils.py`: Embedding utilities (no alignment)
- `run_llm_agent.py`: **CLI entrypoint for LLM agent (recommended)**
- `run_hypothesis_agent.py`: CLI entrypoint for HypothesisAgent
- `create_dummy_graph.py`: Script to create and save dummy graphs

## Notes

- **Pre-computed Embeddings**: Embeddings must be computed and aligned BEFORE using reasoning_layer
- **No Alignment Work**: The reasoning_layer assumes embeddings are already aligned
- **Ready Nodes**: Works with pre-computed nodes and edges (graph traversal only)
- Swap mocks with real STATE/CAPTAIN models by editing `engine/tools_*.py`
- Graph structure allows for future extension to multi-hop reasoning
- All tools use deterministic seeds for reproducible outputs (when using mocks)

## Workflow

### Complete LLM-Powered Workflow

1. **Pre-compute embeddings** (outside reasoning_layer):
   - Compute perturbation, RNA, and protein embeddings
   - Align embeddings across modalities
   - Save aligned embeddings to `/home/nebius/cellian/outputs/cell_embeddings.pt`

2. **Pre-compute graphs** (optional):
   - Create graphs with nodes and edges
   - Save graphs to `/home/nebius/cellian/outputs/graphs/`
   - Or use dummy graphs for testing: `python create_dummy_graph.py JAK1`

3. **Ask natural language question**:
   ```bash
   python run_llm_agent.py "What happens if I knock down JAK1?"
   ```

4. **LLM Agent processes**:
   - Extracts perturbation information from query
   - Executes 3-node chain (HypothesisAgent)
   - Generates LLM interpretation of results
   - Returns comprehensive analysis

### Workflow Steps

```
User Query: "What happens if I knock down JAK1?"
    ↓
LLM extracts: target="JAK1", type="KD"
    ↓
HypothesisAgent.execute():
  - Load pre-computed aligned embeddings
  - Get perturbation embedding (Node 1)
  - Predict RNA (Node 2)
  - Predict protein (Node 3)
  - Compare to ground truth
  - Calculate error propagation
    ↓
LLM interprets results:
  - Summarizes hypothesis
  - Explains validation scores
  - Provides biological insights
    ↓
Return: JSON with query, results, and LLM interpretation
```

## Graph Modes

### 1. Pre-computed Graph
```python
from engine.graph_loader import load_graph
graph = load_graph(filepath="/path/to/graph.json")
agent = HypothesisAgent(registry, data_loader, graph=graph)
```

### 2. Dummy Graph
```python
agent = HypothesisAgent(registry, data_loader, use_dummy_graph=True)
# or from command line: python run_hypothesis_agent.py JAK1 --dummy
```

### 3. Dynamic Graph (Default)
```python
agent = HypothesisAgent(registry, data_loader)
# Graph is built as agent executes
```

