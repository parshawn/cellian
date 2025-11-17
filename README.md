# Cellian: Multi-Omics Hypothesis Engine

<div align="center">

![Cellian Logo](cellian_logo.png)

**A Virtual Cell Platform for Predicting and Analyzing Multi-Omics Perturbations**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

</div>

## Overview

Cellian is an integrated platform that combines deep learning models, pathway analysis, and AI-powered reasoning to predict and analyze the effects of genetic and drug perturbations on cellular systems. The system provides end-to-end predictions from perturbations → RNA → protein, with comprehensive pathway enrichment analysis and hypothesis generation.

### Key Capabilities

- **Gene Perturbations**: Predict RNA and protein changes from CRISPR knockouts, knockouts, and overexpression
- **Drug Perturbations**: Analyze drug effects on cellular transcriptomics and proteomics
- **Dual Perturbations**: Compare gene and drug perturbations side-by-side
- **Pathway Analysis**: Comprehensive GSEA, KEGG, Reactome, and GO enrichment analysis
- **3D Visualizations**: Interactive 3D cell and network visualizations
- **AI-Powered Reasoning**: LLM-based query processing and hypothesis generation

## Features

### 🔬 Multi-Omics Pipeline
- **Perturbation → RNA**: Uses STATE model for gene perturbations and ST-Tahoe for drug perturbations
- **RNA → Protein**: Leverages scTranslator for protein expression prediction
- **Evaluation**: R², Pearson correlation, RMSE, and MAE metrics against ground truth data

### 📊 Analysis & Visualization
- **Differential Expression Analysis**: RNA and protein level comparisons
- **Pathway Enrichment**: KEGG, Reactome, and Gene Ontology analysis
- **Gene Set Enrichment Analysis (GSEA)**: Pathway-level insights
- **Interactive Dashboards**: Real-time results visualization with plots and tables

### 🤖 AI Integration
- **Natural Language Queries**: Ask questions in plain English
- **Intelligent Parsing**: Automatically detects gene/drug perturbations from queries
- **Hypothesis Generation**: AI-powered reasoning about cellular mechanisms
- **Workflow Orchestration**: Automated pipeline execution based on user queries

### 🎨 User Interface
- **3D Cell Visualization**: Watch perturbations inject into cells with animations
- **3D Network Graph**: Interactive translator network showing data flow
- **Real-time Progress**: Live updates during pipeline execution
- **Results Dashboard**: Comprehensive view of all analysis results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Chatbot    │  │  3D Visuals  │  │  Dashboard   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│                   Backend API (FastAPI)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  LLM Layer  │  │ Agent Tools  │  │  Workflow    │     │
│  │  (Gemini)    │  │  Pipeline    │  │ Orchestrator │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│              Agent Tools (Perturbation Pipeline)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   STATE     │  │ scTranslator │  │   Pathway    │     │
│  │  (RNA)      │  │  (Protein)   │  │  Analysis    │     │
│  └─────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 18.x or higher (for frontend)
- **CUDA**: 11.8+ (for GPU acceleration)
- **Models**: 
  - STATE model checkpoint
  - ST-Tahoe model
  - scTranslator checkpoint

### Backend Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd cellian
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install backend-specific dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

5. **Set up environment variables**:
```bash
# Create .env file in backend/ or backend/llm/
cp backend/.env.example backend/.env
# Edit backend/.env and add:
# GOOGLE_API_KEY=your_gemini_api_key_here
```

### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
# or
yarn install
```

3. **Start development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Model Setup

Ensure you have the following models and data files:

- **STATE Model**: Place in `/home/nebius/state/test_replogle/hepg2_holdout/`
- **ST-Tahoe Model**: Place in `/home/nebius/ST-Tahoe/`
- **scTranslator Checkpoint**: Place in `/home/nebius/scTranslator/checkpoint/`
- **Data Files**: Place in `data/perturb-cite-seq/`

Update paths in `Agent_Tools/perturbation_pipeline.py` if using different locations.

## Configuration

### Environment Variables

Create `.env` files in the following locations:

**`backend/.env`** or **`backend/llm/.env`**:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Alternative:
GEMINI_API_KEY=your_gemini_api_key_here
```

### Model Paths

Default model paths (can be overridden via command-line arguments):

- STATE model: `/home/nebius/state/test_replogle/hepg2_holdout/`
- ST-Tahoe: `/home/nebius/ST-Tahoe/`
- scTranslator: `/home/nebius/scTranslator/checkpoint/expression_fine-tuned_scTranslator.pt`

### Data Paths

Default data paths:

- Control template: `data/perturb-cite-seq/scp1064_control_template.h5ad`
- Ground truth RNA: `data/perturb-cite-seq/RNA_expression_combined_mapped.h5ad`
- Ground truth protein: `data/perturb-cite-seq/protein_expression_mapped.h5ad`

## Usage

### Starting the Application

1. **Start the backend**:
```bash
cd backend
python api.py
# Or use the startup script:
./start_backend.sh
```

The backend API will be available at `http://localhost:8000`

2. **Start the frontend** (in a separate terminal):
```bash
cd frontend
npm run dev
```

3. **Open your browser**:
Navigate to `http://localhost:5173`

### Using the Web Interface

1. **Ask a Question**: Type a natural language query in the chatbot:
   - Gene perturbation: `"What happens if I knock down TP53?"`
   - Drug perturbation: `"Dimethyl fumarate"`
   - Both: `"CHCHD2 vs Dimethyl fumarate"`

2. **Select Condition**: Choose experimental condition (Control, IFNγ, or Co-culture)

3. **Watch the Pipeline**: 
   - 3D cell visualization shows perturbation injection
   - 3D network graph highlights active pipeline stages
   - Reasoning log shows real-time progress

4. **View Results**: Check the "Results & Dashboard" tab for:
   - Differential expression analysis
   - Pathway enrichment results
   - Generated hypotheses

### Command-Line Usage

#### Gene Perturbation

```bash
cd Agent_Tools
python perturbation_pipeline.py --target-gene TP53
```

#### Drug Perturbation

```bash
python perturbation_pipeline.py --perturbation-type drug --drug "Dimethyl fumarate"
```

#### With Custom Paths

```bash
python perturbation_pipeline.py \
  --target-gene ACTB \
  --state-model-dir /path/to/state/model \
  --sctranslator-checkpoint /path/to/sctranslator.pt \
  --output-dir /path/to/output
```

See `Agent_Tools/README.md` for detailed command-line options.

## Project Structure

```
cellian/
├── backend/                 # FastAPI backend service
│   ├── api.py              # Main API endpoints
│   ├── llm/                # LLM integration (Gemini)
│   │   ├── input.py        # Query parsing
│   │   └── output.py      # Result interpretation
│   └── Agent_Tools/        # Pipeline tools (symlinked)
│
├── frontend/                # React + TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Chatbot.tsx
│   │   │   ├── Cell3D.tsx
│   │   │   ├── TranslatorNetwork3D.tsx
│   │   │   └── Dashboard.tsx
│   │   └── pages/
│   │       └── Index.tsx   # Main page
│   └── package.json
│
├── Agent_Tools/             # Core perturbation pipeline
│   ├── perturbation_pipeline.py  # Main pipeline orchestrator
│   ├── state_inference.py        # Gene → RNA prediction
│   ├── drug_inference.py         # Drug → RNA prediction
│   ├── sctranslator_inference.py # RNA → Protein prediction
│   ├── pathway_analysis.py       # Pathway enrichment
│   └── evaluation.py             # Metrics calculation
│
├── llm/                     # LLM reasoning layer
│   ├── hypothesis_agent.py  # Hypothesis generation
│   └── perturbation_orchestrator.py
│
├── reasoning_layer/          # Graph-based reasoning
│   └── engine/              # Reasoning engine
│
├── data/                    # Data files (not in git)
│   └── perturb-cite-seq/   # Perturb-CITE-seq dataset
│
├── logs/                    # Workflow logs
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## API Documentation

### Endpoints

#### `POST /api/query/process`
Process a natural language query and extract perturbation information.

**Request**:
```json
{
  "query": "What happens if I knock down TP53?"
}
```

**Response**:
```json
{
  "success": true,
  "perturbation_info": {
    "target": "TP53",
    "type": "KD",
    "has_both": false,
    "confidence": 0.9
  }
}
```

#### `POST /api/workflow/start`
Start a perturbation workflow.

**Request**:
```json
{
  "perturbation_info": {
    "target": "TP53",
    "type": "KD"
  },
  "condition": "Control",
  "perturbation_type": "gene"
}
```

**Response**:
```json
{
  "workflow_id": "uuid-here",
  "status": "pending"
}
```

#### `GET /api/workflow/{workflow_id}/status`
Get workflow status and results.

**Response**:
```json
{
  "status": "running",
  "progress": 0.65,
  "current_step": "Predicting protein changes...",
  "pipeline_stage": "protein",
  "logs": [...],
  "results": {...}
}
```

#### `GET /api/workflow/{workflow_id}/logs`
Get workflow logs (streaming).

### Interactive API Docs

When the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

### Backend Development

```bash
cd backend
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Style

- **Python**: Follow PEP 8, use `black` and `ruff`
- **TypeScript**: Follow ESLint rules, use Prettier

```bash
# Format Python code
black backend/
ruff check backend/

# Format TypeScript code
cd frontend
npm run lint
```

## Troubleshooting

### Common Issues

1. **LLM API Key Not Found**
   - Ensure `GOOGLE_API_KEY` is set in `backend/.env` or `backend/llm/.env`
   - Verify the API key is valid and has proper permissions

2. **Model Files Not Found**
   - Check model paths in `Agent_Tools/perturbation_pipeline.py`
   - Ensure all required checkpoint files exist

3. **Port Already in Use**
   - Backend: Change port in `api.py` or use `--port` flag
   - Frontend: Change port in `vite.config.ts`

4. **CORS Errors**
   - Ensure backend CORS settings include your frontend URL
   - Check `backend/api.py` CORS middleware configuration

5. **Pipeline Fails**
   - Check logs in `logs/workflow_*.log`
   - Verify data files exist and are accessible
   - Ensure all dependencies are installed

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Follow existing code style
- Ensure all tests pass before submitting

## License

[Add your license here]

## Citation

If you use Cellian in your research, please cite:

```bibtex
@software{cellian2024,
  title = {Cellian: Multi-Omics Hypothesis Engine},
  author = {[Your Name/Team]},
  year = {2024},
  url = {https://github.com/yourusername/cellian}
}
```

## Acknowledgments

- **STATE Model**: For gene perturbation → RNA prediction
- **ST-Tahoe**: For drug perturbation → RNA prediction
- **scTranslator**: For RNA → protein translation
- **Gemini API**: For natural language processing
- **React Three Fiber**: For 3D visualizations

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/cellian/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cellian/discussions)

---

<div align="center">

**Built with ❤️ for the scientific community**

*Let's build the virtual cell together!*

</div>
