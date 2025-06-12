# AgentEval: Corporate Governance Scorecard with Agentic AI

A sophisticated multi-agent AI system for automated corporate governance evaluation using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## 🎯 Overview

AgentEval transforms corporate governance assessment through a specialized five-agent architecture that processes unstructured corporate documents to generate standardized governance scores with detailed justifications and source citations.

## ✨ Key Features

- **Multi-Agent Architecture**: Specialized agents for input validation, question generation, research, output validation, and scoring
- **Hybrid Retrieval System**: Combines BM25 keyword search with semantic embeddings for comprehensive document analysis
- **Pre-Computed Optimization**: 20-100x speed improvement through cached chunks and embeddings
- **PDF Slice Reconstruction**: Preserves document structure including tables and charts
- **Intelligent Fallback**: Progressive escalation when retrieval methods find insufficient information
- **Quality Guardrails**: Deterministic validation and hallucination detection
- **Explainable Scoring**: Detailed justifications with preserved source citations

## 🏗️ Architecture

```
                          ┌─────────────────┐
                          │ Input Guardrail │
                          │ Agent           │
                          └─────────┬───────┘
                                    │ validates topic
                                    ▼
                          ┌─────────────────┐
                          │ Question Agent  │◄─────────┐
                          │                 │          │
                          └─────────┬───────┘          │
                                    │ generates question│ generates follow-up
                                    ▼                  │ if gaps found
                          ┌─────────────────┐          │
                          │ Research Agent  │          │
                          │                 │          │
                          └─────────┬───────┘          │
                                    │ produces answer   │
                                    ▼                  │
                          ┌─────────────────┐          │
                          │ Output Guardrail│          │
                          │ Agent           │          │
                          └─────────┬───────┘          │
                                    │ validates answer  │
                                    ▼                  │
                          ┌─────────────────┐          │
                          │ Evidence Pool   │          │
                          │                 │          │
                          └─────────┬───────┘          │
                                    │ check completeness│
                                    └──────────────────┘
                                    │ if sufficient evidence
                                    ▼
                          ┌─────────────────┐
                          │ Scoring Agent   │
                          │                 │
                          └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google API Key (for Gemini models) or Ollama (for local models)
- Required packages: `langchain`, `chromadb`, `sentence-transformers`, `streamlit`

### Installation

```bash
git clone https://github.com/your-username/agentevaluation
cd agentevaluation
pip install -r requirements.txt
```

### Environment Setup

```bash
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Directory Structure

```
./data/COMPANY_NAME/
├── 98_data/           # Place PDF documents here
├── 97_cache/          # Auto-generated cache files
└── 96_results/        # Evaluation results
```

### Basic Usage

```python
from main import OptimizedConfig, OptimizedAgenticOrchestrator, TopicDefinition

# Configure system
config = OptimizedConfig("YOUR_COMPANY")
orchestrator = OptimizedAgenticOrchestrator(config)

# Define evaluation topic
topic = TopicDefinition(
    topic_name="Board Independence",
    goal="Assess board director permanence",
    guidance="Analyze appointment dates...",
    scoring_rubric={
        "0": "Permanent directors present",
        "1": "Permanent directors are lender reps",
        "2": "All directors non-permanent"
    }
)

# Run evaluation
result = orchestrator.evaluate_topic(topic)
print(f"Score: {result['scoring']['score']}/2")
```

### Web Interface

```bash
streamlit run app.py
```

## 📊 Usage Examples

### Command Line Interface

```bash
# Test all retrieval methods
python main.py --test-all

# Test specific method
python main.py --method hybrid

# Performance comparison
python main.py --test-performance
```

### Programmatic Configuration

```python
# Customize retrieval method
config.retrieval_method = "hybrid"  # "bm25", "vector", "direct"

# Configure agent-specific LLMs
config.agent_llms = {
    "research_agent": "gemini-1.5-pro",
    "scoring_agent": "gemini-1.5-flash"
}

# Set agent temperatures
config.agent_temperatures = {
    "input_agent": 0.1,     # Consistent validation
    "research_agent": 0.2,  # Factual analysis
    "scoring_agent": 0.1    # Deterministic scoring
}
```

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `retrieval_method` | Search strategy | `"hybrid"` |
| `max_iterations` | Research cycles | `3` |
| `use_pdf_slices` | PDF reconstruction | `True` |
| `chunk_size` | Text chunk size | `1000` |
| `temperature` | LLM creativity | `0.2` |

## 📈 Performance Metrics

- **Speed**: 20-100x faster than traditional approaches
- **Accuracy**: High fidelity against expert assessments
- **Consistency**: Deterministic scoring with low variance
- **Coverage**: Comprehensive source identification and citation

## 🛡️ Quality Assurance

- **Input Validation**: Topic definition completeness and coherence
- **Output Validation**: Answer quality, citations, and confidence assessment
- **Fallback Mechanisms**: Progressive document escalation when needed
- **Error Handling**: Graceful degradation and comprehensive logging

## 📁 Project Structure

```
├── main.py              # Core agentic system
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── data/               # Document storage
    └── COMPANY/
        ├── 98_data/    # Input PDFs
        ├── 97_cache/   # Cached processing
        └── 96_results/ # Evaluation outputs
```

## 🔄 Agent Workflow

### Iterative Research Process

1. **Input Validation**: Validates topic definition structure and semantics
2. **Initial Question**: Generates strategic question based on rubric analysis
3. **Research Loop** (up to 3 iterations):
   - Research Agent analyzes documents using hybrid retrieval
   - Output Guardrail validates answer quality
   - Question Agent checks if more information needed
   - If gaps found, generates follow-up question
   - Loop continues until sufficient evidence or max iterations
4. **Final Scoring**: Synthesizes all evidence for final score and justification

### Agent Specializations

- **Input Guardrail**: Rule-based + LLM validation (Temperature: 0.1)
- **Question Agent**: Strategic question generation (Temperature: 0.3)
- **Research Agent**: Document analysis with fallback mechanisms (Temperature: 0.2)
- **Output Guardrail**: Deterministic quality validation (No LLM)
- **Scoring Agent**: Evidence synthesis and scoring (Temperature: 0.1)

## 🧪 Testing

```bash
# Quick test with default settings
python main.py

# Test specific company data
python main.py --company COMPANY_NAME

# Benchmark different methods
python main.py --test-all

# Performance analysis
python main.py --test-performance
```

## 🔧 Advanced Configuration

### Custom LLM Models

```python
config.agent_llms = {
    "input_agent": "gemini-1.5-flash",
    "question_agent": "gemini-1.5-flash", 
    "research_agent": "gemini-1.5-pro",
    "scoring_agent": "gemini-1.5-flash"
}
```

### Retrieval Method Selection

- **hybrid**: Best balance of accuracy and coverage (default)
- **bm25**: Fast keyword-based search
- **vector**: Semantic similarity search
- **direct**: Full document analysis (slower but comprehensive)

### Performance Tuning

```python
config.max_iterations = 3           # Research cycles
config.chunk_size = 1000           # Text chunk size
config.chunk_overlap = 200         # Overlap between chunks
config.max_pdf_size_mb = 20        # PDF size limit for direct processing
config.force_recompute = False     # Clear cache and recompute
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

```bibtex
@article{shah2024agentevaluation,
  title={AgentEval: Corporate Governance Scorecard with Agentic AI},
  author={Shah, Monil and Jadhav, Supriya},
  journal={Northwestern University},
  year={2024}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/agentevaluation/issues)
- **Documentation**: See inline code documentation
- **Contact**: {Monilshah2025, SupriyaJadhav2025}@u.northwestern.edu

## 🏆 Research Highlights

- **First multi-agent system** specifically designed for corporate governance evaluation
- **20-100x performance improvement** through pre-computation optimization
- **Hybrid retrieval innovation** combining keyword and semantic search
- **Quality assurance framework** with dedicated guardrail agents
- **Explainable AI** with detailed justifications and source citations

---

**Built with ❤️ by Northwestern University Research Team**
