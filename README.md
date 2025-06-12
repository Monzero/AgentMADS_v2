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
