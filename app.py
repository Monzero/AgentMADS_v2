"""
Streamlit App for Enterprise Agentic Corporate Governance System
================================================================

A comprehensive web interface that provides:
- Company selection
- Custom topic creation
- Method configuration
- Results visualization
- Download capabilities
"""

import streamlit as st
import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sys

# Import your main agentic system
try:
    from main import (
        OptimizedConfig, TopicDefinition, OptimizedAgenticOrchestrator,
        save_results, save_summary_csv
    )
except ImportError:
    st.error("Could not import the main agentic system. Please ensure main.py is in the same directory.")
    st.stop()

def load_sample_topics():
    """Load sample topics from JSON file for Streamlit app"""
    try:
        json_path = "sample_topics.json"
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(__file__), "sample_topics.json")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert string keys to integers
        topics_data = {}
        for key, value in data["topics"].items():
            topics_data[int(key)] = value
            
        return topics_data
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Could not load predefined topics: {e}")
        return {}

# Page configuration
st.set_page_config(
    page_title="Corporate Governance AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .config-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stAlert > div {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    .status-running {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-complete {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Ready"

def get_available_companies() -> List[str]:
    """Get list of available companies based on data directories"""
    data_dir = "./data"
    companies = []
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            company_path = os.path.join(data_dir, item)
            if os.path.isdir(company_path):
                # Check if it has the required structure
                data_subdir = os.path.join(company_path, "98_data")
                if os.path.exists(data_subdir):
                    # Check if there are PDF files
                    pdf_files = [f for f in os.listdir(data_subdir) if f.endswith('.pdf')]
                    if pdf_files:
                        companies.append(item)
    
    return sorted(companies)

def create_topic_form() -> TopicDefinition:
    """Create form for topic definition with predefined options"""
    st.subheader("Define Your Topic")
    
    # Load predefined topics
    predefined_topics = load_sample_topics()
    
    # Create dropdown options
    topic_options = ["Custom (Create your own topic)"]
    for num, topic_data in predefined_topics.items():
        topic_options.append(f"{num}. {topic_data['topic_name']}")
    
    # Topic selection dropdown (only if topics loaded successfully)
    if predefined_topics:
        st.markdown("### Topic Selection")
        selected_option = st.selectbox(
            "Choose a predefined topic or create a custom one:",
            topic_options,
            help="Select from predefined corporate governance topics or create your own custom topic"
        )
    else:
        selected_option = "Custom (Create your own topic)"
        st.info("Creating custom topic (predefined topics not available)")
    
    # Initialize default values
    default_topic_name = "e.g Board Independence"
    default_goal = "To assess if the board have directors with permanent board seats..."
    default_guidance = """ What to look for, what specific computation should be done, which information is key..."""
    default_score_0 = "...condition for giving score 0."
    default_score_1 = "...condition for giving score 1."
    default_score_2 = "...condition for giving score 2."
    
    # Check if a predefined topic was selected
    if selected_option != "Custom (Create your own topic)" and predefined_topics:
        try:
            # Extract topic number from selection
            topic_num = int(selected_option.split(".")[0])
            topic_data = predefined_topics[topic_num]
            
            # Set default values from predefined topic
            default_topic_name = topic_data["topic_name"]
            default_goal = topic_data["goal"]
            default_guidance = topic_data["guidance"]
            default_score_0 = topic_data["scoring_rubric"]["0"]
            default_score_1 = topic_data["scoring_rubric"]["1"]
            default_score_2 = topic_data["scoring_rubric"]["2"]
            
            # Show info about predefined topic
            st.info(f"📋 **Selected Topic:** {topic_data['topic_name']}")
            st.info("💡 **Tip:** You can modify any field below if needed, or keep the predefined values.")
            
        except (ValueError, KeyError, IndexError):
            st.warning("Error loading selected topic. Using default values.")
    
    # Rest of the form remains EXACTLY the same as your original
    with st.form("topic_form"):
        topic_name = st.text_input(
            "Topic Name",
            value=default_topic_name,
            help="A concise name for your evaluation topic"
        )
        
        goal = st.text_area(
            "Goal",
            value=default_goal,
            help="What you want to evaluate or measure",
            height=100
        )
        
        guidance = st.text_area(
            "Guidance",
            value=default_guidance,
            help="Detailed instructions on how to evaluate this topic",
            height=200
        )
        
        st.write("**Scoring Rubric**")
        col1, col2 = st.columns(2)
        
        with col1:
            score_0 = st.text_area(
                "Score 0 (Poor)",
                value=default_score_0,
                help="Criteria for the lowest score"
            )
        
        with col2:
            score_2 = st.text_area(
                "Score 2 (Excellent)",
                value=default_score_2,
                help="Criteria for the highest score"
            )
        
        score_1 = st.text_area(
            "Score 1 (Good)",
            value=default_score_1,
            help="Criteria for the middle score"
        )
        
        submit_topic = st.form_submit_button("Create Topic", type="primary")
        
        if submit_topic:
            if all([topic_name, goal, guidance, score_0, score_1, score_2]):
                return TopicDefinition(
                    topic_name=topic_name,
                    goal=goal,
                    guidance=guidance,
                    scoring_rubric={
                        "0": score_0,
                        "1": score_1,
                        "2": score_2
                    }
                )
            else:
                st.error("Please fill in all fields")
                return None
    
    return None

def create_configuration_panel() -> Dict[str, Any]:
    """Create configuration panel with agent-wise LLM and temperature selection"""
    #st.subheader("Settings")
    
    # Basic settings
    col1, col2 = st.columns(2)
    
    with col1:
        retrieval_method = st.selectbox(
            "Analysis Method",
            ["hybrid", "bm25", "vector"],
            index=0,
            help="Choose the analysis method"
        )
        
        max_iterations = st.number_input(
            "Max Iterations",
            1, 5, 3, 1,
            help="Maximum research iterations"
        )
    
    with col2:
        # force_recompute = st.checkbox(
        #     "Clear Cache & Recompute",
        #     value=False,
        #     help="Force recomputation of all data"
        # )
        
        # if st.button("Clear All Caches", help="Remove all cached data"):
        #     clear_all_caches()
        #     st.success("All caches cleared!")
            
    # Create sub-columns to push elements right
        cache_col1, cache_col2 = st.columns([1, 2])  # 1:2 ratio pushes content right
        
        with cache_col2:  # Right column
            force_recompute = st.checkbox(
                "Clear Cache & Recompute",
                value=False,
                help="Force recomputation of all data"
            )
            
            if st.button("Clear All Caches", help="Remove all cached data"):
                clear_all_caches()
                st.success("All caches cleared!")
                
    # Agent-wise LLM and Temperature Configuration
    st.markdown("---")
    
    with st.expander("Agent Configuration (Advanced)", expanded=False):
        st.subheader("Agent LLM & Temperature Settings")
        st.info("Configure LLM and temperature for each agent. Lower temperature = more consistent, higher = more creative.")
        
        # Available LLM options
        gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        ollama_models = ["llama3", "llama3.1", "mistral", "codellama"]
        all_models = gemini_models + ollama_models
        
        # Default temperatures for each agent type
        default_temps = {
            "input_agent": 0.1,
            "question_agent": 0.3,
            "research_agent": 0.2,
            "scoring_agent": 0.1
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input & Question Agents**")
            
            # Input Agent
            st.markdown("*Input Validation Agent*")
            input_col1, input_col2 = st.columns([2, 1])
            with input_col1:
                input_agent_llm = st.selectbox(
                    "Model",
                    all_models,
                    index=0,
                    help="Validates topic definitions",
                    key="input_llm"
                )
            with input_col2:
                input_agent_temp = st.slider(
                    "Temp",
                    0.0, 1.0, default_temps["input_agent"], 0.1,
                    help="Temperature for input validation",
                    key="input_temp"
                )
            
            # Question Agent
            st.markdown("*Question Generation Agent*")
            question_col1, question_col2 = st.columns([2, 1])
            with question_col1:
                question_agent_llm = st.selectbox(
                    "Model",
                    all_models,
                    index=0,
                    help="Generates research questions",
                    key="question_llm"
                )
            with question_col2:
                question_agent_temp = st.slider(
                    "Temp",
                    0.0, 1.0, default_temps["question_agent"], 0.1,
                    help="Temperature for question generation",
                    key="question_temp"
                )
        
        with col2:
            st.markdown("**Research & Scoring Agents**")
            
            # Research Agent
            st.markdown("*Research Agent*")
            research_col1, research_col2 = st.columns([2, 1])
            with research_col1:
                research_agent_llm = st.selectbox(
                    "Model",
                    all_models,
                    index=1,  # Default to gemini-1.5-pro
                    help="Analyzes documents and extracts information",
                    key="research_llm"
                )
            with research_col2:
                research_agent_temp = st.slider(
                    "Temp",
                    0.0, 1.0, default_temps["research_agent"], 0.1,
                    help="Temperature for document analysis",
                    key="research_temp"
                )
            
            # Scoring Agent
            st.markdown("*Scoring Agent*")
            scoring_col1, scoring_col2 = st.columns([2, 1])
            with scoring_col1:
                scoring_agent_llm = st.selectbox(
                    "Model",
                    all_models,
                    index=0,
                    help="Provides final scoring and justification",
                    key="scoring_llm"
                )
            with scoring_col2:
                scoring_agent_temp = st.slider(
                    "Temp",
                    0.0, 1.0, default_temps["scoring_agent"], 0.1,
                    help="Temperature for final scoring",
                    key="scoring_temp"
                )
        
        # Temperature Recommendations
        st.markdown("---")
        st.markdown("**💡 Temperature Recommendations:**")
        st.markdown("""
        - **Input Validation (0.1)**: Very consistent validation of requirements
        - **Question Generation (0.3)**: Structured but slightly varied questions  
        - **Research Analysis (0.2)**: Factual analysis with minimal creativity
        - **Final Scoring (0.1)**: Highly consistent scoring against rubrics
        
        **Guidelines:** 0.0-0.2 = Consistent | 0.3-0.5 = Balanced | 0.6-1.0 = Creative
        """)
        
        # Show current API status
        st.markdown("---")
        st.markdown("**📡 API Status & Model Info:**")
        google_api_available = bool(os.environ.get("GOOGLE_API_KEY"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**API Status:**")
            if google_api_available:
                st.success("✅ Google API Key configured")
                st.info("Gemini models available")
            else:
                st.warning("⚠️ No Google API Key found")
                st.warning("Only Ollama models will work")
        
        with col2:
            st.markdown("**Selected Models:**")
            selected_models = {
                input_agent_llm, question_agent_llm, 
                research_agent_llm, scoring_agent_llm
            }
            
            gemini_count = len([m for m in selected_models if m.startswith('gemini')])
            ollama_count = len([m for m in selected_models if not m.startswith('gemini')])
            
            if gemini_count > 0:
                st.info(f"🌩️ {gemini_count} Gemini models selected")
            if ollama_count > 0:
                st.info(f"🏠 {ollama_count} Local models selected")
    
    return {
        "retrieval_method": retrieval_method,
        "max_iterations": max_iterations,
        "force_recompute": force_recompute,
        "agent_llms": {
            "input_agent": input_agent_llm,
            "question_agent": question_agent_llm,
            "research_agent": research_agent_llm,
            "scoring_agent": scoring_agent_llm
        },
        "agent_temperatures": {
            "input_agent": input_agent_temp,
            "question_agent": question_agent_temp,
            "research_agent": research_agent_temp,
            "scoring_agent": scoring_agent_temp
        }
    }

def clear_all_caches():
    """Clear all cache files"""
    try:
        companies = get_available_companies()
        for company in companies:
            cache_dir = f"./data/{company}/97_cache/"
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        st.error(f"Error clearing caches: {e}")

def update_status(status: str):
    """Update the current status"""
    st.session_state.current_status = status

def show_status():
    """Display current status"""
    status = st.session_state.current_status
    
    if status == "Ready":
        st.info(f"Status: {status}")
    elif "Running" in status or "Processing" in status:
        st.markdown(f'<div class="status-running">Status: {status}</div>', unsafe_allow_html=True)
    elif "Complete" in status:
        st.markdown(f'<div class="status-complete">Status: {status}</div>', unsafe_allow_html=True)
    elif "Error" in status or "Failed" in status:
        st.markdown(f'<div class="status-error">Status: {status}</div>', unsafe_allow_html=True)
    else:
        st.info(f"Status: {status}")

def run_evaluation(company: str, topic: TopicDefinition, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run the agentic evaluation with agent-specific LLM configuration"""
    
    try:
        # Create configuration
        config = OptimizedConfig(company)
        
        # Apply configuration settings
        config.retrieval_method = config_dict["retrieval_method"]
        config.use_pdf_slices = True
        config.auto_fallback_to_direct = True
        
        # Use sensible defaults for technical parameters
        config.bm25_weight = 0.4
        config.vector_weight = 0.6
        config.similarity_threshold = 0.1
        config.page_buffer = 1
        config.max_pdf_size_mb = 20
        config.max_direct_documents = 3
        config.temperature = config_dict.get("temperature", 0.2)
        config.force_recompute = config_dict.get("force_recompute", False)
        
        # Apply agent-specific LLM and temperature configuration
        if "agent_llms" in config_dict:
            config.agent_llms = config_dict["agent_llms"]
        if "agent_temperatures" in config_dict:
            config.agent_temperatures = config_dict["agent_temperatures"]
        
        # Create orchestrator with agent-specific LLMs
        update_status("Initializing system with custom LLM configuration...")
        orchestrator = OptimizedAgenticOrchestrator(config)
        orchestrator.max_iterations = config_dict["max_iterations"]
        
        # Run evaluation
        update_status("Running evaluation...")
        result = orchestrator.evaluate_topic(topic)

          
        # Save results
        if result and result.get("success", False):
            update_status("Saving results...")
            save_results(result, config)
            #save_summary_csv(result, config) #save_results will anyways call save_summary_csv
            update_status("Evaluation completed successfully")
        else:
            update_status("Evaluation failed")
        
        return result or {"success": False, "error": "No result returned"}
        
    except Exception as e:
        import traceback
        error_msg = f"Error during evaluation: {str(e)}\n{traceback.format_exc()}"
        update_status(f"Error: {str(e)}")
        return {"success": False, "error": str(e)}

def display_results(result: Dict[str, Any]):
    """Display evaluation results with metrics and visualizations"""
    
    if not result.get("success", False):
        st.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")
        return
    
    st.subheader("Evaluation Results")
    
    # Key Metrics
    #col1, col2, col3, col4 = st.columns(4)
    col1,  col3, col4 = st.columns(3)
    
    with col1:
        st.metric(
            "Final Score",
            f"{result['scoring']['score']}/2",
            help="Score based on the defined rubric"
        )
    
    # with col2:
    #     st.metric(
    #         "Confidence",
    #         result['scoring']['confidence'].title(),
    #         help="System confidence in the result"
    #     )
    
    with col3:
        st.metric(
            "Sources Used",
            result['research_summary']['total_sources'],
            help="Number of unique sources referenced"
        )
    
    with col4:
        st.metric(
            "Research Time",
            f"{result['performance_metrics']['research_time']:.2f}s",
            help="Time spent on document research"
        )
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric(
            "Total Time",
            f"{result['performance_metrics']['total_time']:.2f}s"
        )
    
    with perf_col2:
        st.metric(
            "Iterations",
            result['research_summary']['iterations']
        )
    
    with perf_col3:
        st.metric(
            "Questions Asked",
            result['research_summary']['questions_asked']
        )
    
    # Research Summary
    st.subheader("Research Summary")
    
    research_data = {
        "Metric": [
            "Retrieval Method",
            "PDF Slices Used",
            "Optimization Enabled",
            "Answers Approved",
            "Evidence Quality"
        ],
        "Value": [
            result['research_summary']['retrieval_method'],
            "Yes" if result['research_summary']['pdf_slices_used'] else "No",
            "Yes" if result['research_summary']['optimization_enabled'] else "No",
            f"{result['research_summary']['answers_approved']}/{result['research_summary']['questions_asked']}",
            result['scoring']['evidence_quality'].title()
        ]
    }
    
    research_df = pd.DataFrame(research_data)
    st.table(research_df)
    
    # Evidence Details
    st.subheader("Evidence Collected")
    
    for i, evidence in enumerate(result['evidence'], 1):
        with st.expander(f"Evidence {i}: {evidence['question'][:100]}..."):
            st.write("**Question:**")
            st.write(evidence['question'])
            
            st.write("**Answer:**")
            st.write(evidence['answer'])
            
            st.write("**Sources:**")
            for source in evidence['sources']:
                st.write(f"• {source}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Confidence:** {evidence['confidence']}")
            with col2:
                st.write(f"**Has Citations:** {'Yes' if evidence['has_citations'] else 'No'}")
    
    # Final Justification
    st.subheader("Final Justification")
    st.write(result['scoring']['justification'])
    
    # Key Findings
    if result['scoring'].get('key_findings'):
        st.subheader("Key Findings")
        for finding in result['scoring']['key_findings']:
            st.write(f"• {finding}")

def create_history_visualization():
    """Create visualizations for evaluation history"""
    
    if not st.session_state.evaluation_history:
        st.info("No evaluation history yet. Run some evaluations to see trends!")
        return
    
    df = pd.DataFrame(st.session_state.evaluation_history)
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        score_counts = df['final_score'].value_counts().sort_index()
        fig_scores = px.bar(
            x=score_counts.index,
            y=score_counts.values,
            labels={'x': 'Score', 'y': 'Count'},
            title="Distribution of Final Scores"
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.subheader("Performance Over Time")
        fig_time = px.line(
            df,
            x='timestamp',
            y='total_time',
            title="Evaluation Time Trend",
            labels={'total_time': 'Time (seconds)', 'timestamp': 'Date'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Method comparison
    if len(df['retrieval_method'].unique()) > 1:
        st.subheader("Method Performance Comparison")
        method_perf = df.groupby('retrieval_method').agg({
            'final_score': 'mean',
            'total_time': 'mean',
            'unique_sources': 'mean'
        }).round(2)
        
        st.dataframe(method_perf, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("<h1 class='main-header'>Corporate Governance AI</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Control Panel")
        
        # Company Selection
        st.subheader("Select Company")
        companies = get_available_companies()
        
        if not companies:
            st.error("No companies found! Please ensure you have data directories with PDF files.")
            st.stop()
        
        selected_company = st.selectbox(
            "Available Companies",
            companies,
            help="Select a company to analyze"
        )
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Home", "Create Topic", "Run Evaluation", "Results", "History"],
            help="Choose what you want to do"
        )
    
    if page == "Home":
        st.subheader("Welcome to the Corporate Governance AI System")
        
        # Introduction
        st.markdown("""
        <div style="background-color: #f0f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #2d3748; margin-top: 0;">Transform Corporate Governance Analysis with AI</h3>
        <p style="font-size: 1.1em; color: #4a5568; margin-bottom: 0;">
        This enterprise-grade system leverages multiple AI agents to automatically evaluate corporate governance topics 
        by analyzing annual reports, governance documents, and regulatory filings. Get objective, evidence-based 
        assessments in minutes instead of hours.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features
        st.markdown("## **Core Capabilities**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### **Intelligent Document Analysis**
            - **Multi-modal Processing**: Analyzes PDFs with text extraction and visual recognition
            - **Advanced Search**: Combines BM25 keyword search with semantic vector search
            - **Smart Fallback**: Automatically switches methods when initial approaches don't find sufficient information
            - **Source Citation**: Every finding is linked back to specific page numbers and documents
            
            ### **Multi-Agent Workflow**
            - **Input Validation**: Ensures topic definitions are well-formed and evaluable
            - **Strategic Question Generation**: Creates targeted research questions based on scoring rubrics
            - **Iterative Research**: Adapts follow-up questions based on evidence gaps
            - **Quality Assurance**: Validates answer quality before final scoring
            """)
        
        with col2:
            st.markdown("""
            ### **Optimized Performance**
            - **Pre-computed Indexes**: Documents processed once, queried instantly
            - **Intelligent Caching**: Speeds up repeated analyses
            - **Progressive Escalation**: Tries efficient methods first, escalates as needed
            - **Real-time Monitoring**: Track evaluation progress and performance metrics
            
            ### **Comprehensive Analysis**
            - **Evidence-based Scoring**: Uses predefined rubrics for consistent evaluation
            - **Confidence Assessment**: Rates the reliability of findings
            - **Multi-source Validation**: Cross-references information across documents
            - **Audit Trail**: Complete transparency of reasoning and sources
            """)
        
        # How It Works
        st.markdown("## **How It Works**")
        
        st.markdown("""
        <div style="background-color: #fff5f5; padding: 15px; border-left: 5px solid #e53e3e; margin: 10px 0;">
        <strong>Step 1: Topic Definition</strong><br>
        Define what you want to evaluate (e.g., "Board Independence") with clear scoring criteria (0-2 scale)
        </div>
        
        <div style="background-color: #fffaf0; padding: 15px; border-left: 5px solid #dd6b20; margin: 10px 0;">
        <strong>Step 2: Strategic Research</strong><br>
        AI agents generate targeted questions and systematically search through corporate documents
        </div>
        
        <div style="background-color: #f0fff4; padding: 15px; border-left: 5px solid #38a169; margin: 10px 0;">
        <strong>Step 3: Evidence Collection</strong><br>
        System gathers relevant information with source citations and assesses answer quality
        </div>
        
        <div style="background-color: #f7fafc; padding: 15px; border-left: 5px solid #4299e1; margin: 10px 0;">
        <strong>Step 4: Objective Scoring</strong><br>
        Final evaluation against your rubric with detailed justification and confidence assessment
        </div>
        """, unsafe_allow_html=True)
        
        # Use Cases
        st.markdown("## **Perfect For**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Investment Analysis**
            - Due diligence on portfolio companies
            - ESG compliance assessment  
            - Risk factor evaluation
            - Comparative governance analysis
            """)
        
        with col2:
            st.markdown("""
            **Regulatory Compliance**
            - Audit preparation and review
            - Compliance gap identification
            - Best practice benchmarking
            - Policy effectiveness assessment
            """)
        
        with col3:
            st.markdown("""
            **Academic Research**
            - Corporate governance studies
            - Large-scale data analysis
            - Longitudinal trend analysis
            - Cross-industry comparisons
            """)
        
        # Sample Topics
        st.markdown("## **Pre-built Evaluation Topics**")
        
        topics_info = {
            "Board Independence": "Assess director tenure and independence from management influence",
            "AGM Timeliness": "Evaluate how quickly companies hold AGMs after financial year-end",
            "POSH Compliance": "Review sexual harassment prevention policies and incident reporting", 
            "Related Party Oversight": "Analyze governance of related party transaction approval processes",
            "Workforce Diversity": "Measure women's representation across organizational levels"
        }
        
        for topic, description in topics_info.items():
            st.markdown(f"**{topic}**: {description}")
        
        # Getting Started
        st.markdown("## **Getting Started**")
        
        st.markdown("""
        1. **Select Company**: Choose from available company data in the sidebar
        2. **Create Topic**: Pick a pre-built topic or define your own evaluation criteria  
        3. **Configure**: Choose analysis method and agent settings (optional)
        4. **Run Evaluation**: Let the AI agents analyze the documents
        5. **Review Results**: Examine findings, evidence, and download reports
        """)
        
        # Technical Highlights
        with st.expander("**Technical Architecture** (For Technical Users)", expanded=False):
            st.markdown("""
            **Agent Architecture:**
            - **Input Guardrail Agent**: Validates topic definitions using rule-based logic
            - **Question Agent**: Generates strategic research questions using LLM analysis
            - **Research Agent**: Conducts document search with hybrid retrieval methods
            - **Output Guardrail Agent**: Validates answer quality and source citations
            - **Scoring Agent**: Provides final evaluation against defined rubrics
            
            **Retrieval Methods:**
            - **BM25**: Traditional keyword-based search for exact term matching
            - **Vector Search**: Semantic similarity using sentence transformers
            - **Hybrid**: Combines both approaches with weighted scoring
            - **Direct Processing**: Sends entire documents to multimodal AI when needed
            
            **Performance Optimizations:**
            - Pre-computed document chunks and embeddings
            - Intelligent caching with automatic invalidation
            - Progressive document escalation strategies
            - Memory-efficient vector operations
            """)
        
        # Quick stats and company info
        if selected_company:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                data_path = f"./data/{selected_company}/98_data"
                if os.path.exists(data_path):
                    try:
                        doc_count = len([f for f in os.listdir(data_path) if f.endswith('.pdf')])
                        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
                        total_size = sum(os.path.getsize(os.path.join(data_path, f)) for f in pdf_files)
                        total_size_mb = total_size / (1024 * 1024)
                    except Exception:
                        doc_count = 0
                        total_size_mb = 0
                else:
                    doc_count = 0
                    total_size_mb = 0
                
                st.metric("Selected Company", selected_company)
            
            with col2:
                st.metric("Documents Available", f"{doc_count} PDFs")
            
            with col3:
                st.metric("Total Data Size", f"{total_size_mb:.1f} MB")
            
            # Show available documents
            if doc_count > 0:
                with st.expander(f"View Available Documents ({doc_count} files)", expanded=False):
                    try:
                        for file in sorted(pdf_files):
                            file_path = os.path.join(data_path, file)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            st.write(f"**{file}** ({file_size:.1f} MB)")
                    except Exception as e:
                        st.write(f"Could not list files: {e}")
            
            st.info(f"💡 **Ready to analyze {selected_company}!** Navigate to 'Create Topic' to define your evaluation criteria.")
        else:
            st.warning("⚠️ No company selected. Please choose a company from the sidebar to begin.")
            
        # Footer with tips
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #edf2f7; padding: 15px; border-radius: 8px; text-align: center;">
        <h4 style="color: #2d3748; margin-top: 0;">💡 Pro Tips</h4>
        <p style="color: #4a5568; margin-bottom: 0;">
        • Start with pre-built topics to understand the system • Use hybrid retrieval for best results • 
        Check source citations for verification • Download results for further analysis • 
        Clear caches if you encounter processing issues
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Create Topic":
        topic = create_topic_form()
        if topic:
            st.session_state.current_topic = topic
            st.success("Topic created successfully! Go to 'Run Evaluation' to analyze it.")
    
    elif page == "Run Evaluation":
        # Check if we have topic
        if 'current_topic' not in st.session_state:
            st.warning("Please create a topic first!")
            st.stop()
        
        st.subheader("Run Evaluation")
        
        # Configuration section
        st.subheader("Settings")
        config_dict = create_configuration_panel()
        st.session_state.current_config = config_dict
        
        # Show current settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Topic:**")
            st.write(f"Topic: {st.session_state.current_topic.topic_name}")
            st.write(f"Goal: {st.session_state.current_topic.goal[:100]}...")
        
        with col2:
            st.write("**Current Configuration:**")
            if 'current_config' in st.session_state:
                st.write(f"Method: {st.session_state.current_config['retrieval_method']}")
                st.write(f"Max Iterations: {st.session_state.current_config['max_iterations']}")
                
                # Show agent LLM and temperature configuration
                # if 'agent_llms' in st.session_state.current_config:
                #     st.write("**Agent Configuration:**")
                #     agent_llms = st.session_state.current_config['agent_llms']
                #     agent_temps = st.session_state.current_config.get('agent_temperatures', {})
                    
                #     for agent_name, model in agent_llms.items():
                #         temp = agent_temps.get(agent_name, 0.2)
                #         agent_display = agent_name.replace('_', ' ').title()
                #         st.write(f"• {agent_display}: {model} (T:{temp})")
        
        # Status display
        show_status()
        
        # Run button
        if st.button("Start Evaluation", type="primary"):
            if 'current_config' not in st.session_state:
                st.error("Please configure the settings first!")
                st.stop()
                
            with st.spinner("Running evaluation..."):
                
                # Run evaluation
                try:
                    result = run_evaluation(
                        selected_company,
                        st.session_state.current_topic,
                        st.session_state.current_config
                    )
                    
                    if result and result.get("success", False):
                        # Store results
                        st.session_state.evaluation_results = result
                        
                        # Add to history
                        history_entry = {
                            'timestamp': result['timestamp'],
                            'company': selected_company,
                            'topic_name': result['topic']['name'],
                            'final_score': result['scoring']['score'],
                            'confidence': result['scoring']['confidence'],
                            'retrieval_method': result['research_summary']['retrieval_method'],
                            'total_time': result['performance_metrics']['total_time'],
                            'unique_sources': result['research_summary']['total_sources']
                        }
                        st.session_state.evaluation_history.append(history_entry)
                        
                        st.success("Evaluation completed! Check the 'Results' tab to see detailed findings.")
                    else:
                        st.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
    
    elif page == "Results":
        if 'evaluation_results' not in st.session_state or not st.session_state.evaluation_results:
            st.warning("No results available. Please run an evaluation first!")
        else:
            display_results(st.session_state.evaluation_results)
            
            # Download options
            st.subheader("Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download JSON
                json_data = json.dumps(st.session_state.evaluation_results, indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Download summary CSV
                if st.session_state.evaluation_history:
                    df = pd.DataFrame(st.session_state.evaluation_history)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    elif page == "History":
        st.subheader("Evaluation History")
        create_history_visualization()
        
        # Show detailed history table
        if st.session_state.evaluation_history:
            st.subheader("Detailed History")
            df = pd.DataFrame(st.session_state.evaluation_history)
            st.dataframe(df, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History", type="secondary"):
                st.session_state.evaluation_history = []
                st.success("History cleared!")
                st.rerun()

if __name__ == "__main__":
    main()