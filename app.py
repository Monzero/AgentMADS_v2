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
    """Create form for topic definition"""
    st.subheader("Define Your Topic")
    
    with st.form("topic_form"):
        topic_name = st.text_input(
            "Topic Name",
            value="Board Independence",
            help="A concise name for your evaluation topic"
        )
        
        goal = st.text_area(
            "Goal",
            value="To assess if the board have directors with permanent board seats",
            help="What you want to evaluate or measure",
            height=100
        )
        
        guidance = st.text_area(
            "Guidance",
            value="""You need to look for the corporate governance report. Find the reappointment date for each board members. If the reappointment date is either not provided or older than 5 years (i.e some date before 2019), then you need to check appointment date. If appointment date is also older than 5 years (i.e before 2019), mark that board member as permanent. Give list of board members and whether or not they are permanent. In other words, either of appointment date or reappointment date should be within last 5 years. For example, if a board member has appointment date '02-07-2020' and reappointment date is not present, then because the appointment date is within last 5 years (i.e March 2020 to March 2025 assuming we are checking for annual report as of 31st March 2025) then we would label them as 'Not permanent'. Second example, if any board member has appointment date as 01-01-2012 and reappointment date not present, then we would mark them permanent. Do not present output in table format. Give me text based paragraphs. You are looking at the corporate governance report as of 31st March 2024. Make sure you quote this source in the answer with the page number from which you extract the information.""",
            help="Detailed instructions on how to evaluate this topic",
            height=200
        )
        
        st.write("**Scoring Rubric**")
        col1, col2 = st.columns(2)
        
        with col1:
            score_0 = st.text_area(
                "Score 0 (Poor)",
                value="if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders.",
                help="Criteria for the lowest score"
            )
        
        with col2:
            score_2 = st.text_area(
                "Score 2 (Excellent)",
                value="if All directors are marked as non-permanent board members",
                help="Criteria for the highest score"
            )
        
        score_1 = st.text_area(
            "Score 1 (Good)",
            value="if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so.",
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
        config.bm25_weight = 0.5
        config.vector_weight = 0.5
        config.similarity_threshold = 0.001
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
        print(f"Evaluation result: {result}")
        update_status("Evaluation complete...")
        
        
        # Save results
        if result and result.get("success", False):
            update_status("Saving results...")
            save_results(result, config)
            save_summary_csv(result, config)
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
    
    # Main content area
    if page == "Home":
        st.subheader("Welcome to the Corporate Governance AI System")
        
        st.markdown("""
        This system uses advanced AI agents to evaluate corporate governance topics by:
        
        **Intelligent Document Analysis**
        - Advanced search combining multiple methods
        - Automatic PDF processing
        - Smart fallback mechanisms
        
        **Multi-Agent Workflow**
        - Input validation and guardrails
        - Dynamic question generation
        - Research with source verification
        - Automated scoring against custom rubrics
        
        **Optimized Performance**
        - Pre-computed embeddings and indexes
        - Intelligent caching system
        - Real-time status updates
        
        **Comprehensive Analysis**
        - Evidence-based scoring
        - Source citations and verification
        - Performance metrics and insights
        """)
        
        # Quick stats
        if selected_company:
            data_path = f"./data/{selected_company}/98_data"
            if os.path.exists(data_path):
                doc_count = len([f for f in os.listdir(data_path) if f.endswith('.pdf')])
            else:
                doc_count = 0
            st.info(f"Selected Company: **{selected_company}** with **{doc_count} documents**")
    
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