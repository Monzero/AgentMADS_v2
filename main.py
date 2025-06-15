import os
import json
import logging
import time
import fitz
import tempfile
import base64
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# LangChain imports for enterprise retrieval
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging with colors for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Color codes for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title: str, content: str = "", color: str = Colors.OKBLUE):
    """Print a clearly visible section with color coding"""
    print(f"\n{color}{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}{Colors.ENDC}")
    if content:
        print(f"{content}")
    print()

def print_agent_action(agent_name: str, action: str, details: str = "", llm_model: str = ""):
    """Print agent actions with LLM information"""
    llm_info = f" (LLM: {llm_model})" if llm_model else ""
    print(f"\n{Colors.OKCYAN}🤖 {agent_name.upper()} AGENT{llm_info}{Colors.ENDC}")
    print(f"   Action: {action}")
    if details:
        print(f"   Details: {details}")

def print_llm_interaction(prompt_type: str, prompt: str, response: str, truncate: bool = True):
    """Print LLM interactions with clear formatting"""
    print(f"\n{Colors.WARNING}💭 LLM INTERACTION: {prompt_type}{Colors.ENDC}")
    
    # Truncate long prompts for readability
    if truncate and len(prompt) > 500:
        prompt_display = prompt[:500] + "\n... [truncated] ..."
    else:
        prompt_display = prompt
    
    print(f"{Colors.OKGREEN}📝 PROMPT:{Colors.ENDC}")
    print(f"   {prompt_display}")
    
    print(f"{Colors.OKGREEN}🎯 RESPONSE:{Colors.ENDC}")
    if truncate and len(response) > 800:
        response_display = response[:800] + "\n... [truncated] ..."
    else:
        response_display = response
    print(f"   {response_display}")

def print_retrieval_info(method: str, query: str, chunks_found: int, sources: List[str]):
    """Print retrieval process information"""
    print(f"\n{Colors.HEADER}🔎 RETRIEVAL PROCESS{Colors.ENDC}")
    print(f"   Method: {method}")
    print(f"   Query: {query}")
    print(f"   Chunks Found: {chunks_found}")
    print(f"   Sources: {', '.join(sources[:5])}{'...' if len(sources) > 5 else ''}")

@dataclass
class TopicDefinition:
    """User-defined topic with goal, guidance, and rubric"""
    topic_name: str
    goal: str
    guidance: str
    scoring_rubric: Dict[str, str]

@dataclass
class Question:
    """A research question with metadata"""
    text: str
    purpose: str
    priority: str

@dataclass
class Answer:
    """Answer with source citations"""
    question: str
    answer: str
    sources: List[str]
    confidence: str
    has_citations: bool

@dataclass
class PrecomputedChunk:
    """A precomputed chunk with all necessary data"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""


class LLMManager:
    
    def __init__(self, config):
        self.config = config
        self.llm_cache = {}  # Cache LLM instances
        self._setup_api()
    
    def _setup_api(self):
        """Setup API keys"""
        from dotenv import load_dotenv
        load_dotenv()
        
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                self.genai_client = genai
                print_section("API SETUP", f"✅ Google API configured successfully", Colors.OKGREEN)
            except ImportError:
                logger.error("Google Generative AI package not installed")
                self.genai_client = None
        else:
            logger.warning("No Google API key found")
            self.genai_client = None
    
    def get_llm(self, agent_name: str, model_name: str = None):
        """Get LLM instance for specific agent"""
        
        # Determine which model to use
        if model_name:
            selected_model = model_name
        elif hasattr(self.config, 'agent_llms') and agent_name in self.config.agent_llms:
            selected_model = self.config.agent_llms[agent_name]
        else:
            # Fallback to default
            selected_model = self.config.gemini_model if self.genai_client else self.config.ollama_model
        
        # Check cache first
        cache_key = f"{agent_name}_{selected_model}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key], selected_model
        
        # Create new LLM instance
        try:
            if selected_model.startswith('gemini') and self.genai_client:
                llm = self.genai_client.GenerativeModel(selected_model)
                print(f"   🌩️ {agent_name} using {selected_model} (Gemini)")
            else:
                # Use Ollama for non-Gemini models

                from langchain_ollama import OllamaLLM
                agent_temp = self.config.agent_temperatures.get(agent_name, self.config.temperature)
                llm = OllamaLLM(model=selected_model, temperature=agent_temp)
                print(f"   🏠 {agent_name} using {selected_model} (Ollama)")
            
            # Cache the instance
            self.llm_cache[cache_key] = llm
            return llm, selected_model
            
        except Exception as e:
            logger.error(f"Error creating LLM for {agent_name} with {selected_model}: {e}")
            
            # Fallback to default
            if self.genai_client and not selected_model.startswith('gemini'):
                fallback_model = self.config.gemini_model
                llm = self.genai_client.GenerativeModel(fallback_model)
                print(f"   ⚠️ {agent_name} falling back to {fallback_model}")
            else:                
                from langchain_ollama import OllamaLLM
                fallback_model = self.config.ollama_model
                agent_temp = self.config.agent_temperatures.get(agent_name, self.config.temperature)
                llm = OllamaLLM(model=fallback_model, temperature=agent_temp)
                
                print(f"   ⚠️ {agent_name} falling back to {fallback_model}")
            
            return llm, fallback_model

class OptimizedConfig:
    def __init__(self, company: str, base_path: str = None):
        self.company = company
        self.base_path = base_path or f"./data/{company}/"
        self.data_path = os.path.join(self.base_path, "98_data/")
        self.cache_path = os.path.join(self.base_path, "97_cache/")
        
        # Create cache directory
        os.makedirs(self.cache_path, exist_ok=True)
        
        # Model settings
        self.model_provider = "gemini"
        self.gemini_model = "gemini-1.5-flash"
        self.ollama_model = "llama3"
        self.temperature = 0.2
        
        # Agent-specific LLM configuration (can be overridden)
        self.agent_llms = {
            "input_agent": "gemini-1.5-flash",      # Fast validation
            "question_agent": "gemini-1.5-flash",   # Quick question generation
            "research_agent": "gemini-1.5-pro",     # High-quality analysis
            "scoring_agent": "gemini-1.5-flash"     # Fast scoring
        }
        
        # Agent-specific temperature configuration
        self.agent_temperatures = {
            "input_agent": 0.1,        # Very low - need consistent validation
            "question_agent": 0.3,     # Low-medium - structured questions
            "research_agent": 0.2,     # Low - factual analysis
            "scoring_agent": 0.1       # Very low - consistent scoring
        }
        
        # Retrieval configuration
        self.retrieval_method = "hybrid"
        self.bm25_weight = 0.6
        self.vector_weight = 0.4
        self.max_chunks_for_query = 10  # No limit - use all chunks retriever finds
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # PDF slice configuration
        self.use_pdf_slices = True
        self.page_buffer = 1
        self.max_pdf_size_mb = 20
        
        # Optimization settings
        self.force_recompute = False  # Set to True to force recomputation
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # No artificial limits - let retrievers return all relevant content
        self.similarity_threshold = 0.0  # Minimum similarity score for vector search
        self.bm25_score_threshold = 0.0  # Minimum BM25 score (usually 0 is fine)
        
        # Intelligent fallback settings
        self.auto_fallback_to_direct = True  # Enable automatic fallback to direct method
        self.progressive_escalation = True   # Use progressive document escalation (one-by-one)
        self.max_direct_documents = 3        # Max documents to try in progressive escalation
        self.fallback_keywords = [           # Keywords that indicate insufficient information
            "no information", "not available", "cannot find", "no mention", 
            "not specified", "no details", "insufficient information",
            "not provided", "no relevant information", "not disclosed",
            "no data", "not found", "cannot locate", "no such information"
        ]
        
        # API setup
        self._setup_api()
    
    def _setup_api(self):
        """Setup API keys"""
        from dotenv import load_dotenv
        load_dotenv()
        
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                self.genai_client = genai
                print_section("API SETUP", f"✅ Google API configured successfully", Colors.OKGREEN)
            except ImportError:
                logger.error("Google Generative AI package not installed")
                self.genai_client = None
        else:
            logger.warning("No Google API key found")
            self.genai_client = None

class OptimizedDocumentProcessor:
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.embeddings_model = None
        self.page_chunks = {}  # filename -> List[PrecomputedChunk]
        self.text_chunks = {}  # filename -> List[PrecomputedChunk]
        self.bm25_retrievers = {}  # filename -> BM25Retriever
        self.vector_stores = {}  # filename -> Chroma
        self.hybrid_retrievers = {}  # filename -> EnsembleRetriever
        
        

        # Initialize everything once
        self._initialize_embeddings()
        self._precompute_all_chunks()
        self._create_all_retrievers()
        
        print_section("OPTIMIZATION COMPLETE", 
                     f"Pre-computed chunks for {len(self.page_chunks)} documents\n" +
                     f"All retrievers ready for fast queries", Colors.OKGREEN)
    
    def _initialize_embeddings(self):
        """Initialize embeddings model once"""
        print_section("INITIALIZING EMBEDDINGS MODEL")
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            
            
            print(f"   ✅ Embeddings model loaded: {self.config.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings_model = None
    
    def _get_cache_path(self, document_path: str, cache_type: str) -> str:
        """Get cache file path for a document"""
        doc_name = os.path.basename(document_path).replace('.pdf', '')
        return os.path.join(self.config.cache_path, f"{doc_name}_{cache_type}.pkl")
    
    def _is_cache_valid(self, document_path: str, cache_path: str) -> bool:
        """Check if cache is valid (newer than source document)"""
        if self.config.force_recompute:
            return False
        
        if not os.path.exists(cache_path):
            return False
        
        doc_mtime = os.path.getmtime(document_path)
        cache_mtime = os.path.getmtime(cache_path)
        
        return cache_mtime > doc_mtime
    
    def _save_to_cache(self, data: Any, cache_path: str):
        """Save data to cache file"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"   💾 Cached to: {os.path.basename(cache_path)}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_path: str) -> Any:
        """Load data from cache file with error handling"""
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"   📂 Loaded from cache: {os.path.basename(cache_path)}")
            return data
        except Exception as e:
            print(f"   ⚠️ Cache load failed for {os.path.basename(cache_path)}: {e}")
            print(f"   🔄 Will recompute from scratch")
            # Delete the corrupted cache file
            try:
                os.remove(cache_path)
                print(f"   🗑️ Removed corrupted cache file")
            except:
                pass
            return None
    
    def _precompute_all_chunks(self):
        """Pre-compute all chunks for all documents"""
        print_section("PRE-COMPUTING ALL CHUNKS")
        
        if not os.path.exists(self.config.data_path):
            logger.warning(f"Data path does not exist: {self.config.data_path}")
            return
        
        pdf_files = [f for f in os.listdir(self.config.data_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.config.data_path, pdf_file)
            print(f"\n   📄 Processing {pdf_file}...")
            
            # Check if PDF is readable
            try:
                test_doc = fitz.open(pdf_path)
                if len(test_doc) == 0:
                    print(f"   ⚠️ {pdf_file} is empty - skipping")
                    test_doc.close()
                    continue
                test_doc.close()
            except Exception as e:
                print(f"   ❌ {pdf_file} is corrupted or unreadable: {e}")
                continue
            
            # Pre-compute page-level chunks
            page_cache_path = self._get_cache_path(pdf_path, "page_chunks")
            if self._is_cache_valid(pdf_path, page_cache_path):
                cached_page_chunks = self._load_from_cache(page_cache_path)
                if cached_page_chunks is not None:
                    self.page_chunks[pdf_file] = cached_page_chunks
                else:
                    # Cache loading failed, recompute
                    page_chunks = self._create_page_chunks(pdf_path)
                    if page_chunks:
                        self.page_chunks[pdf_file] = page_chunks
                        self._save_to_cache(page_chunks, page_cache_path)
                    else:
                        print(f"   ⚠️ No page chunks created for {pdf_file}")
                        continue
            else:
                page_chunks = self._create_page_chunks(pdf_path)
                if page_chunks:
                    self.page_chunks[pdf_file] = page_chunks
                    self._save_to_cache(page_chunks, page_cache_path)
                else:
                    print(f"   ⚠️ No page chunks created for {pdf_file}")
                    continue
            
            # Pre-compute text chunks with embeddings
            text_cache_path = self._get_cache_path(pdf_path, "text_chunks")
            if self._is_cache_valid(pdf_path, text_cache_path):
                cached_text_chunks = self._load_from_cache(text_cache_path)
                if cached_text_chunks is not None:
                    self.text_chunks[pdf_file] = cached_text_chunks
                else:
                    # Cache loading failed, recompute
                    text_chunks = self._create_text_chunks_with_embeddings(pdf_path)
                    if text_chunks:
                        self.text_chunks[pdf_file] = text_chunks
                        self._save_to_cache(text_chunks, text_cache_path)
                    else:
                        print(f"   ⚠️ No text chunks created for {pdf_file}")
                        continue
            else:
                text_chunks = self._create_text_chunks_with_embeddings(pdf_path)
                if text_chunks:
                    self.text_chunks[pdf_file] = text_chunks
                    self._save_to_cache(text_chunks, text_cache_path)
                else:
                    print(f"   ⚠️ No text chunks created for {pdf_file}")
                    continue
            
            # Only print summary if we have valid chunks
            page_count = len(self.page_chunks.get(pdf_file, []))
            text_count = len(self.text_chunks.get(pdf_file, []))
            print(f"   ✅ {pdf_file}: {page_count} page chunks, {text_count} text chunks")
    
    def _create_page_chunks(self, document_path: str) -> List[PrecomputedChunk]:
        """Create page-level chunks"""
        chunks = []
        try:
            doc = fitz.open(document_path)
            file_name = os.path.basename(document_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    chunk = PrecomputedChunk(
                        content=text,
                        metadata={
                            "source": file_name,
                            "page": page_num + 1,
                            "file_path": document_path,
                            "total_pages": len(doc),
                            "chunk_type": "page"
                        },
                        chunk_id=f"{file_name}_page_{page_num + 1}"
                    )
                    chunks.append(chunk)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error creating page chunks: {e}")
        
        return chunks
    
    def _create_text_chunks_with_embeddings(self, document_path: str) -> List[PrecomputedChunk]:
        """Create text chunks and compute embeddings"""
        chunks = []
        try:
            # Load document
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Convert to PrecomputedChunk with embeddings
            for i, doc in enumerate(split_docs):
                # Compute embedding if model is available
                embedding = None
                if self.embeddings_model:
                    try:
                        embedding = self.embeddings_model.embed_query(doc.page_content)
                        embedding = np.array(embedding)
                    except Exception as e:
                        logger.error(f"Error computing embedding for chunk {i}: {e}")
                
                chunk = PrecomputedChunk(
                    content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "file_path": document_path,
                        "source": os.path.basename(document_path)
                    },
                    embedding=embedding,
                    chunk_id=f"{os.path.basename(document_path)}_chunk_{i}"
                )
                chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {e}")
        
        return chunks
    
    def _create_all_retrievers(self):
        """Create all retrievers once using pre-computed chunks"""
        print_section("CREATING ALL RETRIEVERS")
        
        for pdf_file in self.page_chunks.keys():
            print(f"\n   🔧 Creating retrievers for {pdf_file}...")
            
            # Create BM25 retriever from page chunks
            if self.page_chunks[pdf_file]:
                page_docs = [
                    Document(page_content=chunk.content, metadata=chunk.metadata)
                    for chunk in self.page_chunks[pdf_file]
                ]
                
                bm25_retriever = BM25Retriever.from_documents(page_docs)
                if max_chunks := self.config.max_chunks_for_query:
                    bm25_retriever.k = max_chunks
                self.bm25_retrievers[pdf_file] = bm25_retriever
                print(f"   ✅ BM25 retriever created")
            
            # Create vector store from text chunks with pre-computed embeddings
            if self.text_chunks[pdf_file] and self.embeddings_model:
                try:
                    text_docs = [
                        Document(page_content=chunk.content, metadata=chunk.metadata)
                        for chunk in self.text_chunks[pdf_file]
                    ]
                    
                    doc_id = pdf_file.replace('.pdf', '')
                    persist_directory = os.path.join(self.config.cache_path, f"vector_store_{doc_id}")
                    
                    # Create vector store (this still recreates, but only once)
                    vector_store = Chroma.from_documents(
                        documents=text_docs,
                        embedding=self.embeddings_model,
                        persist_directory=persist_directory
                    )
                    
                    self.vector_stores[pdf_file] = vector_store
                    print(f"   ✅ Vector store created")
                    
                except Exception as e:
                    logger.error(f"Error creating vector store: {e}")
            
            # Create hybrid retriever if both are available
            if pdf_file in self.bm25_retrievers and pdf_file in self.vector_stores:
                try:
                    vector_retriever = self.vector_stores[pdf_file].as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": self.config.max_chunks_for_query} if self.config.max_chunks_for_query else {}
                    )
                    
                    hybrid_retriever = EnsembleRetriever(
                        retrievers=[self.bm25_retrievers[pdf_file], vector_retriever],
                        weights=[self.config.bm25_weight, self.config.vector_weight]
                    )
                    
                    self.hybrid_retrievers[pdf_file] = hybrid_retriever
                    print(f"   ✅ Hybrid retriever created")
                    
                except Exception as e:
                    logger.error(f"Error creating hybrid retriever: {e}")
    
    def fast_semantic_search(self, query: str, pdf_file: str, top_k: int = None) -> List[PrecomputedChunk]:
        """Fast semantic search using pre-computed embeddings"""
        if top_k is None and self.config.max_chunks_for_query:
            top_k = self.config.max_chunks_for_query
        
        if pdf_file not in self.text_chunks or not self.embeddings_model:
            return []
        
        try:
            # Compute query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            query_embedding = np.array(query_embedding)
            
            # Calculate similarities with all pre-computed embeddings
            similarities = []
            for chunk in self.text_chunks[pdf_file]:
                if chunk.embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, chunk.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                    )
                    similarities.append((similarity, chunk))
            
            # Sort by similarity and return top_k (or all if top_k is None)
            similarities.sort(key=lambda x: x[0], reverse=True)
            if top_k:
                return [chunk for _, chunk in similarities[:top_k]]
            else:
                return [chunk for _, chunk in similarities]
            
        except Exception as e:
            logger.error(f"Error in fast semantic search: {e}")
            return []
    
    def get_retriever(self, pdf_file: str, method: str = None):
        """Get pre-created retriever for fast queries"""
        if method is None:
            method = self.config.retrieval_method
        
        if method == "bm25":
            return self.bm25_retrievers.get(pdf_file)
        elif method == "vector":
            if pdf_file in self.vector_stores:
                return self.vector_stores[pdf_file].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 100}  # High limit for comprehensive results
                )
        elif method == "hybrid":
            return self.hybrid_retrievers.get(pdf_file)
        elif method == "direct":
            return None  # Direct method doesn't use retrievers
        
        return None

class InputGuardrailAgent:
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM for input validation"""
        self.llm, self.current_model = self.llm_manager.get_llm("input_agent")
    
        
    def validate_topic_definition(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Validate if topic definition is appropriate for evaluation"""
        
        # print_agent_action("INPUT GUARDRAIL", "Validating Topic Definition", 
        #                   f"Topic: {topic.topic_name}")
        
        print_agent_action("INPUT GUARDRAIL", "Validating Topic Definition", 
                  f"Topic: {topic.topic_name}", self.current_model)
        
        # Basic validation first
        basic_issues = []
        
        print(f"   🔍 Checking basic requirements...")
        
        if not topic.topic_name or not topic.topic_name.strip():
            basic_issues.append("Topic name is empty")
        
        if not topic.goal or not topic.goal.strip():
            basic_issues.append("Goal is empty")
        
        if not topic.guidance or not topic.guidance.strip():
            basic_issues.append("Guidance is empty")
        
        if not topic.scoring_rubric or len(topic.scoring_rubric) == 0:
            basic_issues.append("Scoring rubric is empty")
        
        if len(topic.goal.strip()) < 10:
            basic_issues.append("Goal is too short to be meaningful")
        
        if len(topic.guidance.strip()) < 10:
            basic_issues.append("Guidance is too short to be meaningful")
        
        if basic_issues:
            print(f"   ❌ Basic validation failed: {basic_issues}")
            return {
                "valid": False,
                "issues": basic_issues,
                "suggestions": ["Please provide more detailed information for the empty or very short fields"]
            }
        
        print(f"   ✅ Basic validation passed")
        
        if not self.llm:
            print(f"   ⚠️ No LLM available, being permissive")
            return {"valid": True, "issues": [], "suggestions": []}
        
        # Use LLM for more nuanced validation
        prompt = f"""
        You are helping validate a corporate governance topic definition. Be VERY LENIENT and permissive.
        Only mark as invalid if there are SERIOUS, OBVIOUS problems that would make evaluation impossible.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        SCORING RUBRIC: {json.dumps(topic.scoring_rubric, indent=2)}
        
        Validation criteria (ONLY mark invalid for serious issues):
        1. Is the goal clear enough to understand what needs to be evaluated?
        2. Does the guidance give some direction on how to evaluate?
        3. Does the scoring rubric have different levels? (i.e 0,1 and 2)
        4. Does it contain any inappropriate language?
        5. Is user asking things which is taboo or not allowed?
        
        IMPORTANT: Err on the side of marking topics as VALID
        
        Respond in JSON format:
        {{
            "valid": true/false,
            "issues": ["only list SERIOUS issues that prevent evaluation"],
            "suggestions": ["gentle suggestions for improvement, not requirements"]
        }}
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self.llm.invoke(prompt)
            
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                #this if loop disables the LLM validation (so not ideal)
                if not result.get("valid", True) and not basic_issues:
                    print(f"   🔄 LLM marked invalid, but overriding to valid (lenient policy)")
                    result["valid"] = True
                    if result.get("issues"):
                        result["suggestions"] = result.get("suggestions", []) + [f"Consider: {issue}" for issue in result["issues"]]
                        result["issues"] = []
                
                status = "✅ VALID" if result["valid"] else "❌ INVALID"
                print(f"   {status} - Issues: {len(result.get('issues', []))}, Suggestions: {len(result.get('suggestions', []))}")
                
                return result
            else:
                print(f"   ⚠️ Could not parse LLM response, defaulting to valid")
                return {"valid": True, "issues": [], "suggestions": []}
                
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            print(f"   ⚠️ Validation error, defaulting to valid")
            return {"valid": True, "issues": [], "suggestions": []}

class QuestionAgent:
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM for question generation"""
        self.llm, self.current_model = self.llm_manager.get_llm("question_agent")
        
    def generate_initial_question(self, topic: TopicDefinition) -> Question:
        """Generate the first question based on rubric analysis"""
        
        # print_agent_action("QUESTION", "Generating Initial Question", 
        #                   f"Analyzing rubric for: {topic.topic_name}")
        
        print_agent_action("QUESTION", "Generating Initial Question", 
                  f"Analyzing rubric for: {topic.topic_name}", self.current_model)
        
        if not self.llm:
            fallback_q = Question(
                text=f"What information is available about {topic.topic_name}?",
                purpose="Fallback question due to LLM unavailability",
                priority="high"
            )
            print(f"   ⚠️ No LLM available, using fallback question")
            print(f"   ❓ Question: {fallback_q.text}")
            return fallback_q
        
        prompt = f"""
        You are an expert corporate governance analyst. Analyze this topic and create ONE key question that will help distinguish between the scoring levels.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        Your task:
        1. Identify what key information differentiates between score levels
        2. Create ONE specific, document-searchable question that targets this differentiator
        3. The question should be answerable using corporate documents
        
        Respond in JSON format:
        {{
            "question": "Your specific question here",
            "purpose": "Why this question helps distinguish between rubric levels",
            "priority": "high"
        }}
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self.llm.invoke(prompt)
            
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                question = Question(
                    text=result.get("question", ""),
                    purpose=result.get("purpose", ""),
                    priority=result.get("priority", "high")
                )
                
                print(f"   ✅ Generated question: {question.text}")
                print(f"   🎯 Purpose: {question.purpose}")
                
                return question
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            fallback_q = Question(
                text=f"What specific information about {topic.topic_name} is disclosed in the documents?",
                purpose="Fallback question due to parsing error",
                priority="high"
            )
            print(f"   ⚠️ Error occurred, using fallback question")
            print(f"   ❓ Question: {fallback_q.text}")
            return fallback_q
    
    def generate_follow_up_question(self, topic: TopicDefinition, existing_answers: List[Answer]) -> Optional[Question]:
        """Generate follow-up question based on gaps in existing answers"""
        
        # print_agent_action("QUESTION", "Checking for Follow-up Questions", 
        #                   f"Analyzing {len(existing_answers)} existing answers")
        
        print_agent_action("QUESTION", "Checking for Follow-up Questions", 
                  f"Analyzing {len(existing_answers)} existing answers", self.current_model)
        
        if not self.llm:
            print(f"   ⚠️ No LLM available for follow-up questions")
            return None
        
        answer_context = "\n".join([
            f"Q: {ans.question}\nA: {ans.answer[:200]}...\nSources: {', '.join(ans.sources[:3])}"
            for ans in existing_answers
        ])
        
        prompt = f"""
        You are evaluating a corporate governance topic. Based on existing research, determine if you need ONE more question to properly apply the scoring rubric.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        EXISTING RESEARCH:
        {answer_context}
        
        Analysis:
        1. Can you confidently apply the scoring rubric with the existing information?
        2. What specific gap prevents proper scoring?
        3. If a gap exists, what ONE question would fill it?
        
        Respond in JSON format:
        {{
            "needs_more_info": true/false,
            "gap_identified": "description of information gap",
            "question": "specific question to fill the gap (if needed)",
            "purpose": "how this question enables proper scoring",
            "priority": "high/medium/low"
        }}
        
        Only generate a question if it's truly necessary for scoring. Be conservative.
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self.llm.invoke(prompt)
            
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get("needs_more_info", False) and result.get("question"):
                    question = Question(
                        text=result["question"],
                        purpose=result.get("purpose", ""),
                        priority=result.get("priority", "medium")
                    )
                    
                    print(f"   ✅ Follow-up needed: {question.text}")
                    print(f"   🎯 Gap: {result.get('gap_identified', 'Not specified')}")
                    
                    return question
                else:
                    print(f"   ✅ No follow-up needed - sufficient information for scoring")
            
            return None
                
        except Exception as e:
            logger.error(f"Follow-up question generation error: {e}")
            print(f"   ⚠️ Error in follow-up analysis")
            return None

class OptimizedResearchAgent:
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.document_processor = OptimizedDocumentProcessor(config)
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM for document analysis"""
        self.llm, self.current_model = self.llm_manager.get_llm("research_agent")
    
    def find_relevant_chunks_optimized(self, question: str, max_chunks: int = None) -> List[Document]:
        """Find relevant chunks using pre-computed data - MUCH FASTER"""
        # No arbitrary limits - use similarity thresholds instead
        
        # print_agent_action("RESEARCH", "Fast Chunk Retrieval", 
        #                   f"Query: {question[:100]}...")
        
        print_agent_action("RESEARCH", "Fast Chunk Retrieval", 
                  f"Query: {question[:100]}...", "Pre-computed indexes")
        
        
        all_relevant_chunks = []
        
        pdf_files = list(self.document_processor.page_chunks.keys())
        
        for pdf_file in pdf_files:
            print(f"   🚀 Fast retrieval from {pdf_file}...")
            
            if self.config.retrieval_method == "direct":
                print(f"   📋 Direct method - skipping retrieval")
                continue
            
            # Use pre-created retrievers for INSTANT results
            retriever = self.document_processor.get_retriever(pdf_file, self.config.retrieval_method)
            if not retriever:
                print(f"   ⚠️ No retriever available for {pdf_file}")
                print(f"       Reason: Failed to create chunks or embeddings for this document")
                print(f"       Check: Document may be corrupted, empty, or have processing errors")
                continue
            
            try:
                start_time = time.time()
                
                # Fast retrieval using pre-built indexes
                try:
                    relevant_chunks = retriever.invoke(question)
                except (AttributeError, TypeError):
                    relevant_chunks = retriever.get_relevant_documents(question)
                
                retrieval_time = time.time() - start_time
                
                # Filter by similarity if we're using vector or hybrid search
                if self.config.retrieval_method in ["vector", "hybrid"]:
                    filtered_chunks = self._filter_chunks_by_similarity(question, relevant_chunks, pdf_file)
                    print(f"   🔍 Similarity filtering: {len(relevant_chunks)} → {len(filtered_chunks)} chunks")
                    relevant_chunks = filtered_chunks
                
                # Add metadata
                for chunk in relevant_chunks:
                    if 'file_path' not in chunk.metadata:
                        chunk.metadata['file_path'] = os.path.join(self.config.data_path, pdf_file)
                
                all_relevant_chunks.extend(relevant_chunks)
                print(f"   ⚡ Found {len(relevant_chunks)} relevant chunks in {retrieval_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Error retrieving from {pdf_file}: {e}")
                print(f"   ❌ Retrieval error for {pdf_file}: {str(e)}")
                continue
        
        # Sort by relevance if we can calculate scores
        if self.config.retrieval_method in ["vector", "hybrid"]:
            all_relevant_chunks = self._sort_chunks_by_relevance(question, all_relevant_chunks)
        
        print(f"   📊 Total chunks found: {len(all_relevant_chunks)} (no artificial limits)")
        
        sources = list(set([chunk.metadata.get('source', 'Unknown') for chunk in all_relevant_chunks]))
        print_retrieval_info(self.config.retrieval_method, question, len(all_relevant_chunks), sources)
        
        return all_relevant_chunks
    
    def _filter_chunks_by_similarity(self, question: str, chunks: List[Document], pdf_file: str) -> List[Document]:
        """Filter chunks based on similarity threshold instead of arbitrary limits"""
        if not self.document_processor.embeddings_model:
            return chunks  # Can't filter without embeddings
        
        try:
            # Get query embedding
            query_embedding = self.document_processor.embeddings_model.embed_query(question)
            query_embedding = np.array(query_embedding)
            
            filtered_chunks = []
            
            for chunk in chunks:
                # Find corresponding precomputed chunk to get embedding
                chunk_text = chunk.page_content
                similarity_score = 0.0
                
                # Find in precomputed chunks
                if pdf_file in self.document_processor.text_chunks:
                    for precomputed_chunk in self.document_processor.text_chunks[pdf_file]:
                        if precomputed_chunk.content == chunk_text and precomputed_chunk.embedding is not None:
                            # Calculate similarity
                            similarity_score = np.dot(query_embedding, precomputed_chunk.embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(precomputed_chunk.embedding)
                            )
                            break
                
                # Apply threshold
                if similarity_score >= self.config.similarity_threshold:
                    # Add similarity score to metadata for sorting
                    chunk.metadata['similarity_score'] = similarity_score
                    filtered_chunks.append(chunk)
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error filtering by similarity: {e}")
            return chunks  # Return original chunks if filtering fails
    
    def _sort_chunks_by_relevance(self, question: str, chunks: List[Document]) -> List[Document]:
        """Sort chunks by relevance score"""
        try:
            # Sort by similarity score if available
            chunks_with_scores = [(chunk.metadata.get('similarity_score', 0.0), chunk) for chunk in chunks]
            chunks_with_scores.sort(key=lambda x: x[0], reverse=True)
            
            return [chunk for score, chunk in chunks_with_scores]
            
        except Exception as e:
            logger.error(f"Error sorting chunks: {e}")
            return chunks
    
    def research_question(self, question: str) -> Answer:
        """Research using optimized pre-computed data with intelligent fallback"""
        
        #print_agent_action("RESEARCH", "Researching Question", question)
        
        print_agent_action("RESEARCH", "Researching Question", question, self.current_model)
        
        if not self.llm:
            print(f"   ❌ No LLM available for research")
            return Answer(
                question=question,
                answer="No LLM available for research",
                sources=[],
                confidence="low",
                has_citations=False
            )
        
        # Always start with retrieval method first
        print(f"   🔍 Phase 1: Trying retrieval-based approach...")
        
        start_time = time.time()
        relevant_chunks = self.find_relevant_chunks_optimized(question)
        retrieval_time = time.time() - start_time
        
        print(f"   ⚡ Retrieval completed in {retrieval_time:.3f}s")
        
        if not relevant_chunks:
            print(f"   ❌ No relevant chunks found - will try direct method")
            return self._fallback_to_direct_method(question, "No relevant chunks found")
        
        # Try retrieval-based approach first
        print(f"   📝 Processing {len(relevant_chunks)} chunks with retrieval method...")
        
        if self.config.use_pdf_slices:
            print(f"   🔀 Using PDF slice reconstruction...")
            answer_text = self._query_with_pdf_slices_and_check(question, relevant_chunks)
        else:
            print(f"   📝 Using text chunks...")
            answer_text = self._query_with_chunks_and_check(question, relevant_chunks)
        
        # Check if LLM found sufficient information
        if self._should_fallback_to_direct(answer_text, question):
            print(f"   🔄 LLM indicates insufficient information - falling back to direct method")
            return self._fallback_to_direct_method(question, "Insufficient information in chunks", relevant_chunks)
        
        # Retrieval was successful
        sources = self._extract_sources_from_chunks(relevant_chunks)
        has_citations = self._has_source_citations(answer_text)
        confidence = self._assess_confidence(answer_text, relevant_chunks)
        
        print(f"   ✅ Research completed with retrieval method - Confidence: {confidence}, Citations: {has_citations}")
        
        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            has_citations=has_citations
        )
    
    def _query_with_chunks_and_check(self, question: str, chunks: List[Document]) -> str:
        """Query with text chunks and explicit relevance checking"""
        
        print(f"   📝 Querying with {len(chunks)} text chunks...")
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk.metadata.get('source', 'Unknown')}, Page: {chunk.metadata.get('page', 'Unknown')}"
            chunk_text = chunk.page_content[:1500]
            context_parts.append(f"[Chunk {i+1}] {source_info}\n{chunk_text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""
        You are analyzing corporate governance documents to answer a specific question.
        
        QUESTION: {question}
        
        DOCUMENT EXCERPTS:
        {context}
        
        Instructions:
        1. Answer the question based ONLY on the provided document excerpts
        2. ALWAYS include specific source citations (page numbers and document names)
        3. If the excerpts do not contain sufficient information to answer the question, explicitly state: "INSUFFICIENT INFORMATION: The provided excerpts do not contain enough details to fully answer this question."
        4. Be precise and factual
        5. Format citations as: "According to [document name], page [number]..."
        
        IMPORTANT: If you cannot find relevant information in the excerpts, clearly state that the information is not available rather than making assumptions.
        
        Provide a comprehensive answer with proper source citations, or clearly indicate if information is insufficient.
        """
        
        print_llm_interaction("Text Chunk Query with Relevance Check", prompt, "", truncate=True)
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                result = response.text
            else:
                result = self.llm.invoke(prompt)
            
            print_llm_interaction("Text Chunk Query with Relevance Check", "", result, truncate=True)
            return result
                
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return f"Error querying documents: {str(e)}"
    
    def _query_with_pdf_slices_and_check(self, question: str, chunks: List[Document]) -> str:
        """Query with PDF slices and explicit relevance checking"""
        try:
            if not self.config.genai_client:
                print(f"   ⚠️ No Gemini client, falling back to text chunks")
                return self._query_with_chunks_and_check(question, chunks)
            
            print(f"   🔀 Reconstructing PDF slices from {len(chunks)} chunks...")
            
            # Extract file paths and page numbers - WITH DEBUGGING
            pdf_slices = []
            
            for i, chunk in enumerate(chunks):
                slice_info = {"page": 1}
                
                if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                    metadata = chunk.metadata
                    
                    # Get file path - try multiple keys
                    file_path = None
                    for key in ['file_path', 'source', 'path']:
                        if key in metadata and metadata[key]:
                            file_path = metadata[key]
                            break
                    
                    if file_path:
                        # Handle file path correctly
                        if os.path.isabs(file_path):
                            final_file_path = file_path
                        elif file_path.startswith('./') or '/' in file_path:
                            final_file_path = file_path
                        else:
                            final_file_path = os.path.join(self.config.data_path, file_path)
                        
                        slice_info["file_path"] = final_file_path
                    
                    # Get page number - try multiple keys
                    page_num = None
                    for key in ['page', 'page_number', 'page_num']:
                        if key in metadata and metadata[key]:
                            try:
                                page_num = int(metadata[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    if page_num:
                        slice_info["page"] = page_num
                
                if "file_path" in slice_info and os.path.exists(slice_info["file_path"]):
                    pdf_slices.append(slice_info)
            
            if not pdf_slices:
                print(f"   ⚠️ No valid PDF slices found, falling back to text chunks")
                return self._query_with_chunks_and_check(question, chunks)
            
            # Group slices by file path and add page buffers
            files_to_pages = {}
            for s in pdf_slices:
                file_path = s['file_path']
                page = int(s['page'])
                
                if file_path not in files_to_pages:
                    files_to_pages[file_path] = set()
                
                # Add page buffers
                for offset in range(-self.config.page_buffer, self.config.page_buffer + 1):
                    buffered_page = page + offset
                    if buffered_page > 0:
                        files_to_pages[file_path].add(buffered_page)
            
            # Create temporary PDF with relevant pages
            output_pdf = fitz.open()
            added_pages = {}
            total_pages_added = 0
            total_pages_attempted = 0
            failed_pages = 0
            
            for file_path, pages in files_to_pages.items():
                try:
                    doc = fitz.open(file_path)
                    total_pages = len(doc)
                    file_name = os.path.basename(file_path)
                    
                    if file_name not in added_pages:
                        added_pages[file_name] = []
                    
                    # Sort pages and filter valid ones
                    valid_pages = []
                    for p in sorted(pages):
                        if 1 <= p <= total_pages:
                            valid_pages.append(p - 1)
                    
                    for i, page_num in enumerate(valid_pages):
                        total_pages_attempted += 1
                        try:
                            if page_num < 0 or page_num >= total_pages:
                                failed_pages += 1
                                continue
                            
                            output_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            added_pages[file_name].append(page_num + 1)
                            total_pages_added += 1
                            
                        except Exception as page_error:
                            failed_pages += 1
                            continue
                    
                    doc.close()
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    try:
                        if os.path.exists(file_path):
                            doc_for_count = fitz.open(file_path)
                            total_file_pages = len(doc_for_count)
                            doc_for_count.close()
                            file_pages = len([p for p in pages if 1 <= p <= total_file_pages])
                        else:
                            file_pages = 0
                    except:
                        file_pages = 0
                    failed_pages += file_pages
                    total_pages_attempted += file_pages
            
            # Calculate success rate
            success_rate = (total_pages_added / total_pages_attempted) if total_pages_attempted > 0 else 0
            
            # Decision logic: when to use PDF vs fall back to text
            if total_pages_added == 0:
                output_pdf.close()
                print(f"   ❌ No pages could be extracted - falling back to text chunks")
                return self._query_with_chunks_and_check(question, chunks)
            elif success_rate < 0.5:
                output_pdf.close()
                print(f"   ⚠️ Low success rate ({success_rate:.1%}) - falling back to text chunks for reliability")
                return self._query_with_chunks_and_check(question, chunks)
            
            # Continue with PDF
            if output_pdf.page_count == 0:
                output_pdf.close()
                return self._query_with_chunks_and_check(question, chunks)
            
            # Save temporary PDF and query Gemini
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"temp_slice_{int(time.time())}.pdf")
            output_pdf.save(temp_pdf_path)
            output_pdf.close()
            
            file_size_mb = os.path.getsize(temp_pdf_path) / (1024 * 1024)
            print(f"   📤 Querying Gemini with reconstructed PDF ({total_pages_added} pages, {file_size_mb:.2f} MB)...")
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            with open(temp_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            enhanced_question = f"""
            {question}
            
            IMPORTANT: If the document does not contain sufficient information to answer this question, please explicitly state: "INSUFFICIENT INFORMATION: This document does not contain enough details to fully answer this question."
            
            Always include specific page numbers in your citations.
            """
            
            print_llm_interaction("PDF Slice Query with Relevance Check", enhanced_question, "", truncate=False)
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    enhanced_question
                ]
            )
            
            result = response.text
            
            print_llm_interaction("PDF Slice Query with Relevance Check", "", result, truncate=True)
            
            # Create page range citations
            citations = []
            for file_name, pages in added_pages.items():
                if not pages:
                    continue
                    
                pages = sorted(list(set(pages)))
                ranges = []
                
                if len(pages) > 0:
                    range_start = pages[0]
                    prev_page = pages[0]
                    
                    for page in pages[1:]:
                        if page > prev_page + 1:
                            if range_start == prev_page:
                                ranges.append(f"{range_start}")
                            else:
                                ranges.append(f"{range_start}-{prev_page}")
                            range_start = page
                        prev_page = page
                    
                    if range_start == prev_page:
                        ranges.append(f"{range_start}")
                    else:
                        ranges.append(f"{range_start}-{prev_page}")
                
                citations.append(f"pp. {', '.join(ranges)} ({file_name})")
            
            if citations:
                result += f"\n\nSources: {'; '.join(citations)}"
            
            # Clean up
            try:
                os.remove(temp_pdf_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary PDF: {cleanup_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in PDF slice query: {e}")
            return self._query_with_chunks_and_check(question, chunks)
    
    def _should_fallback_to_direct(self, answer_text: str, question: str) -> bool:
        """Check if the LLM response indicates insufficient information"""
        
        if not self.config.auto_fallback_to_direct:
            return False
        
        answer_lower = answer_text.lower()
        
        # Check for explicit insufficient information marker
        if "INSUFFICIENT INFORMATION:" in answer_text:
            print(f"   🔍 LLM explicitly indicated insufficient information")
            return True
        
        # Check for common phrases indicating lack of information
        insufficient_indicators = 0
        for keyword in self.config.fallback_keywords:
            if keyword.lower() in answer_lower:
                insufficient_indicators += 1
        
        # If multiple indicators or answer is very short, consider fallback
        if insufficient_indicators >= 2:
            print(f"   🔍 Multiple insufficient information indicators found ({insufficient_indicators})")
            return True
        
        # Check if answer is suspiciously short (likely indicates no information found)
        if len(answer_text.strip()) < 100 and insufficient_indicators >= 1:
            print(f"   🔍 Short answer with insufficient information indicators")
            return True
        
        return False
    
    def _fallback_to_direct_method(self, question: str, reason: str, relevant_chunks: List[Document] = None) -> Answer:
        """Fallback to direct method with progressive document escalation"""
        
        print(f"   🔄 FALLING BACK TO DIRECT METHOD")
        print(f"   📋 Reason: {reason}")
        
        # Identify and rank documents by relevance
        if relevant_chunks:
            # Use documents identified by retrieval with relevance ranking
            relevant_docs = {}
            for chunk in relevant_chunks:
                doc_name = chunk.metadata.get('source', 'Unknown')
                if doc_name not in relevant_docs:
                    relevant_docs[doc_name] = 0
                relevant_docs[doc_name] += 1
            
            # Sort by relevance (chunk count)
            sorted_docs = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"   📊 Document relevance ranking:")
            for doc_name, chunk_count in sorted_docs:
                print(f"     • {doc_name}: {chunk_count} relevant chunks")
        else:
            # Use all available documents if no retrieval info
            pdf_files = [f for f in os.listdir(self.config.data_path) if f.endswith('.pdf')]
            sorted_docs = [(doc_name, 0) for doc_name in pdf_files]
            
            print(f"   📂 Using all available documents (no retrieval ranking):")
            for doc_name, _ in sorted_docs:
                print(f"     • {doc_name}")
        
        # Progressive escalation: try one document at a time
        print(f"   🎯 Starting progressive escalation strategy...")
        
        combined_context = ""
        processed_docs = []
        
        for attempt, (doc_name, chunk_count) in enumerate(sorted_docs[:self.config.max_direct_documents], 1):
            doc_path = os.path.join(self.config.data_path, doc_name)
            
            if not os.path.exists(doc_path):
                print(f"   ❌ Document not found: {doc_name}")
                continue
            
            # Check file size
            file_size_mb = os.path.getsize(doc_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                print(f"   ⚠️ {doc_name} too large ({file_size_mb:.1f} MB), skipping")
                continue
            
            try:
                print(f"   📖 Attempt {attempt}: Processing {doc_name} ({chunk_count} relevant chunks, {file_size_mb:.1f} MB)")
                
                # Enhanced question for progressive attempts
                if attempt == 1:
                    enhanced_question = f"""
{question}

IMPORTANT: Please provide a complete and comprehensive answer. If this document does not contain sufficient information to fully answer the question, explicitly state: "INSUFFICIENT INFORMATION: This document does not contain enough details to fully answer this question."

Include specific page numbers in your citations.
"""
                else:
                    enhanced_question = f"""
{question}

CONTEXT: Previous document(s) provided some information but it was insufficient. This is attempt #{attempt} to find complete information.

IMPORTANT: Please provide a complete and comprehensive answer based on this document. If this document ALSO does not contain sufficient information to fully answer the question, explicitly state: "INSUFFICIENT INFORMATION: This document does not contain enough details to fully answer this question."

Include specific page numbers in your citations.
"""
                
                answer = self._query_entire_document_with_enhanced_question(doc_path, enhanced_question)
                
                if answer:
                    processed_docs.append(doc_name)
                    
                    # Check if this attempt was sufficient
                    if not self._should_fallback_to_direct(answer, question):
                        print(f"   ✅ SUCCESS on attempt {attempt}! {doc_name} provided sufficient information")
                        
                        # Combine with any previous context if this wasn't the first attempt
                        if combined_context:
                            final_answer = f"{combined_context}\n\n=== ADDITIONAL INFORMATION FROM {doc_name} ===\n{answer}"
                        else:
                            final_answer = f"=== FROM {doc_name} ===\n{answer}"
                        
                        return Answer(
                            question=question,
                            answer=final_answer.strip(),
                            sources=processed_docs,
                            confidence="high" if attempt == 1 else "medium",
                            has_citations=self._has_source_citations(answer)
                        )
                    else:
                        print(f"   ⚠️ Attempt {attempt} insufficient, escalating to next document...")
                        # Add this document's info to context for next attempt
                        combined_context += f"\n\n=== FROM {doc_name} ===\n{answer}"
                else:
                    print(f"   ⚠️ No response from {doc_name}")
                    
            except Exception as e:
                logger.error(f"Error processing {doc_name} with direct method: {e}")
                print(f"   ❌ Error processing {doc_name}: {e}")
                continue
        
        # If we get here, all attempts failed
        print(f"   ❌ Progressive escalation failed after {len(processed_docs)} attempts")
        
        if combined_context:
            # Return what we found, even if insufficient
            return Answer(
                question=question,
                answer=f"Partial information found across {len(processed_docs)} documents:\n{combined_context}",
                sources=processed_docs,
                confidence="low",
                has_citations=self._has_source_citations(combined_context)
            )
        else:
            return Answer(
                question=question,
                answer=f"Could not find relevant information using progressive direct method. Reason: {reason}",
                sources=[],
                confidence="low",
                has_citations=False
            )
    
    def _query_entire_document_with_enhanced_question(self, document_path: str, enhanced_question: str) -> str:
        """Query entire document with enhanced question for progressive escalation"""
        try:
            if not self.config.genai_client:
                return None
            
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                print(f"   ⚠️ Document too large: {file_size_mb:.2f} MB > {self.config.max_pdf_size_mb} MB")
                return None
            
            with open(document_path, 'rb') as f:
                pdf_bytes = f.read()
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            print_llm_interaction("Progressive Direct Method Query", enhanced_question, "", truncate=True)
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    enhanced_question
                ]
            )
            
            result = response.text
            print_llm_interaction("Progressive Direct Method Query", "", result, truncate=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying entire document: {e}")
            return None
    
    def _decide_processing_method(self, question: str, chunks: List[Document]) -> str:
        """Intelligently decide between retrieval and direct methods"""
        
        if not self.config.auto_method_selection:
            return self.config.retrieval_method
        
        print(f"   🤖 INTELLIGENT METHOD SELECTION:")
        
        # Analyze chunk distribution by document
        doc_chunk_count = {}
        doc_quality_scores = {}
        
        for chunk in chunks:
            doc_name = chunk.metadata.get('source', 'Unknown')
            if doc_name not in doc_chunk_count:
                doc_chunk_count[doc_name] = 0
                doc_quality_scores[doc_name] = []
            
            doc_chunk_count[doc_name] += 1
            
            # Get quality score if available
            similarity_score = chunk.metadata.get('similarity_score', 0.5)
            doc_quality_scores[doc_name].append(similarity_score)
        
        # Calculate metrics
        total_chunks = len(chunks)
        num_documents = len(doc_chunk_count)
        avg_chunks_per_doc = total_chunks / num_documents if num_documents > 0 else 0
        
        # Calculate average quality scores per document
        doc_avg_quality = {}
        for doc_name, scores in doc_quality_scores.items():
            doc_avg_quality[doc_name] = sum(scores) / len(scores) if scores else 0.5
        
        overall_avg_quality = sum(doc_avg_quality.values()) / len(doc_avg_quality) if doc_avg_quality else 0.5
        
        print(f"       📊 Analysis Results:")
        print(f"         • Total chunks found: {total_chunks}")
        print(f"         • Documents involved: {num_documents}")
        print(f"         • Avg chunks per document: {avg_chunks_per_doc:.1f}")
        print(f"         • Overall quality score: {overall_avg_quality:.2f}")
        
        # Decision logic
        reasons = []
        
        # Factor 1: Too many chunks suggest comprehensive document coverage
        if total_chunks > self.config.direct_method_threshold:
            reasons.append(f"High chunk count ({total_chunks} > {self.config.direct_method_threshold})")
            direct_score = 30
        else:
            direct_score = 0
        
        # Factor 2: High chunks per document suggests broad relevance
        if avg_chunks_per_doc > 15:
            reasons.append(f"High density per document ({avg_chunks_per_doc:.1f} chunks/doc)")
            direct_score += 25
        
        # Factor 3: Low quality chunks suggest retrieval isn't working well
        if overall_avg_quality < self.config.min_chunk_quality_score:
            reasons.append(f"Low chunk quality ({overall_avg_quality:.2f} < {self.config.min_chunk_quality_score})")
            direct_score += 20
        
        # Factor 4: Few documents with many chunks each
        if num_documents <= self.config.max_direct_documents and avg_chunks_per_doc > 10:
            reasons.append(f"Few docs ({num_documents}) with high relevance")
            direct_score += 15
        
        # Factor 5: Check if documents are small enough for direct processing
        direct_eligible_docs = self._check_documents_eligible_for_direct(doc_chunk_count.keys())
        if len(direct_eligible_docs) == num_documents:
            reasons.append("All documents are small enough for direct processing")
            direct_score += 10
        
        print(f"       🎯 Decision Factors:")
        for reason in reasons:
            print(f"         • {reason}")
        
        print(f"       📈 Direct method score: {direct_score}/100")
        
        # Make decision
        if direct_score >= 50:  # Threshold for switching to direct
            selected_method = "direct"
            print(f"       ✅ DECISION: Using DIRECT method (score: {direct_score})")
            print(f"       📋 Will send {len(direct_eligible_docs)} full documents to Gemini")
        else:
            selected_method = "retrieval"
            print(f"       ✅ DECISION: Using RETRIEVAL method (score: {direct_score})")
            print(f"       📝 Will process {total_chunks} chunks with {self.config.retrieval_method}")
        
        return selected_method
    
    def _check_documents_eligible_for_direct(self, doc_names: List[str]) -> List[str]:
        """Check which documents are small enough for direct processing"""
        eligible_docs = []
        
        for doc_name in doc_names:
            doc_path = os.path.join(self.config.data_path, doc_name)
            if os.path.exists(doc_path):
                try:
                    file_size_mb = os.path.getsize(doc_path) / (1024 * 1024)
                    if file_size_mb <= self.config.max_pdf_size_mb:
                        eligible_docs.append(doc_name)
                except Exception as e:
                    logger.error(f"Error checking file size for {doc_name}: {e}")
        
        return eligible_docs
    
    def _research_with_intelligent_direct_method(self, question: str, chunks: List[Document]) -> str:
        """Use direct method on documents identified by retrieval"""
        
        # Identify relevant documents from chunks
        relevant_docs = {}
        for chunk in chunks:
            doc_name = chunk.metadata.get('source', 'Unknown')
            if doc_name not in relevant_docs:
                relevant_docs[doc_name] = 0
            relevant_docs[doc_name] += 1
        
        # Sort documents by relevance (number of chunks)
        sorted_docs = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   📋 Document relevance ranking:")
        for doc_name, chunk_count in sorted_docs:
            print(f"     • {doc_name}: {chunk_count} relevant chunks")
        
        # Process top documents directly
        combined_answer = ""
        all_sources = []
        processed_docs = 0
        
        for doc_name, chunk_count in sorted_docs:
            if processed_docs >= self.config.max_direct_documents:
                print(f"   ⚠️ Reached max direct documents limit ({self.config.max_direct_documents})")
                break
            
            doc_path = os.path.join(self.config.data_path, doc_name)
            
            if not os.path.exists(doc_path):
                print(f"   ❌ Document not found: {doc_name}")
                continue
            
            # Check file size
            file_size_mb = os.path.getsize(doc_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                print(f"   ⚠️ {doc_name} too large ({file_size_mb:.1f} MB), skipping")
                continue
            
            try:
                print(f"   📖 Processing {doc_name} directly (relevance: {chunk_count} chunks, size: {file_size_mb:.1f} MB)...")
                answer = self._query_entire_document(doc_path, question)
                if answer:
                    combined_answer += f"\n\n=== FROM {doc_name} ===\n{answer}"
                    all_sources.append(doc_name)
                    processed_docs += 1
                    print(f"   ✅ Got response from {doc_name}")
                else:
                    print(f"   ⚠️ No response from {doc_name}")
            except Exception as e:
                logger.error(f"Error processing {doc_name} with direct method: {e}")
                print(f"   ❌ Error processing {doc_name}: {e}")
                continue
        
        if not combined_answer:
            print(f"   ❌ Could not process any documents with intelligent direct method")
            return "Could not process documents with intelligent direct method"
        
        print(f"   ✅ Intelligent direct method completed - Processed {processed_docs} documents")
        
        return combined_answer.strip()
    
    def _research_with_direct_method(self, question: str) -> Answer:
        """Research using direct document processing"""
        print(f"   📋 Using direct document processing...")
        
        pdf_files = list(self.document_processor.page_chunks.keys())
        combined_answer = ""
        all_sources = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.config.data_path, pdf_file)
            
            try:
                print(f"   📖 Processing {pdf_file} directly...")
                answer = self._query_entire_document(pdf_path, question)
                if answer:
                    combined_answer += f"\n\nFrom {pdf_file}:\n{answer}"
                    all_sources.append(pdf_file)
                    print(f"   ✅ Got response from {pdf_file}")
                else:
                    print(f"   ⚠️ No response from {pdf_file}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file} with direct method: {e}")
                continue
        
        if not combined_answer:
            print(f"   ❌ Could not process any documents with direct method")
            return Answer(
                question=question,
                answer="Could not process documents with direct method",
                sources=[],
                confidence="low",
                has_citations=False
            )
        
        has_citations = self._has_source_citations(combined_answer)
        confidence = "high" if len(all_sources) > 1 else "medium"
        
        print(f"   ✅ Direct method completed - Sources: {len(all_sources)}, Citations: {has_citations}")
        
        return Answer(
            question=question,
            answer=combined_answer.strip(),
            sources=all_sources,
            confidence=confidence,
            has_citations=has_citations
        )
    
    def _query_entire_document(self, document_path: str, question: str) -> str:
        """Query entire document using Gemini multimodal"""
        try:
            if not self.config.genai_client:
                return None
            
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                print(f"   ⚠️ Document too large: {file_size_mb:.2f} MB > {self.config.max_pdf_size_mb} MB")
                return None
            
            print(f"   📤 Sending {file_size_mb:.2f} MB PDF to Gemini...")
            
            with open(document_path, 'rb') as f:
                pdf_bytes = f.read()
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    question
                ]
            )
            
            print(f"   ✅ Got response from Gemini ({len(response.text)} characters)")
            return response.text
            
        except Exception as e:
            logger.error(f"Error querying entire document: {e}")
            return None
    
    def _query_with_pdf_slices(self, question: str, chunks: List[Document]) -> str:
        """Query with PDF slice reconstruction"""
        try:
            if not self.config.genai_client:
                print(f"   ⚠️ No Gemini client, falling back to text chunks")
                return self._query_with_chunks(question, chunks)
            
            print(f"   🔀 Reconstructing PDF slices from {len(chunks)} chunks...")
            
            # Extract file paths and page numbers - WITH DEBUGGING
            pdf_slices = []
            
            print(f"   🔍 DEBUG: Analyzing chunk metadata...")
            
            for i, chunk in enumerate(chunks):
                slice_info = {"page": 1}
                
                print(f"     Chunk {i+1} metadata: {chunk.metadata}")
                
                if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                    metadata = chunk.metadata
                    
                    # Get file path - try multiple keys
                    file_path = None
                    for key in ['file_path', 'source', 'path']:
                        if key in metadata and metadata[key]:
                            file_path = metadata[key]
                            print(f"     Found file info in '{key}': {file_path}")
                            break
                    
                    if file_path:
                        # Handle file path correctly
                        if os.path.isabs(file_path):
                            # Already absolute path, use as-is
                            final_file_path = file_path
                        elif file_path.startswith('./') or '/' in file_path:
                            # Already a relative path that includes directory, use as-is
                            final_file_path = file_path
                        else:
                            # Just a filename, build full path
                            final_file_path = os.path.join(self.config.data_path, file_path)
                        
                        slice_info["file_path"] = final_file_path
                        print(f"     Final file path: {final_file_path}")
                        print(f"     File exists: {os.path.exists(final_file_path)}")
                    
                    # Get page number - try multiple keys
                    page_num = None
                    for key in ['page', 'page_number', 'page_num']:
                        if key in metadata and metadata[key]:
                            try:
                                page_num = int(metadata[key])
                                print(f"     Found page info in '{key}': {page_num}")
                                break
                            except (ValueError, TypeError):
                                print(f"     Invalid page number in '{key}': {metadata[key]}")
                                pass
                    
                    if page_num:
                        slice_info["page"] = page_num
                
                if "file_path" in slice_info and os.path.exists(slice_info["file_path"]):
                    pdf_slices.append(slice_info)
                    print(f"     ✅ Valid slice: {slice_info}")
                else:
                    print(f"     ❌ Invalid slice: {slice_info}")
            
            print(f"   📊 Found {len(pdf_slices)} valid PDF slices out of {len(chunks)} chunks")
            
            if not pdf_slices:
                print(f"   ⚠️ No valid PDF slices found, falling back to text chunks")
                print(f"   🔍 DEBUG: This usually means:")
                print(f"       • File paths in metadata are incorrect")
                print(f"       • PDF files don't exist at expected locations") 
                print(f"       • Page numbers are missing from metadata")
                return self._query_with_chunks(question, chunks)
            
            # Group slices by file path and add page buffers
            files_to_pages = {}
            for s in pdf_slices:
                file_path = s['file_path']
                page = int(s['page'])
                
                if file_path not in files_to_pages:
                    files_to_pages[file_path] = set()
                
                # Add page buffers
                for offset in range(-self.config.page_buffer, self.config.page_buffer + 1):
                    buffered_page = page + offset
                    if buffered_page > 0:  # Only add positive page numbers
                        files_to_pages[file_path].add(buffered_page)
            
            print(f"   📄 Processing {len(files_to_pages)} files with page buffers...")
            for file_path, pages in files_to_pages.items():
                original_pages = [s['page'] for s in pdf_slices if s['file_path'] == file_path]
                print(f"     {os.path.basename(file_path)}: original pages {sorted(original_pages)} → with buffer {sorted(list(pages))}")
            
            # Create temporary PDF with relevant pages
            output_pdf = fitz.open()
            added_pages = {}
            total_pages_added = 0
            total_pages_attempted = 0
            failed_pages = 0
            
            for file_path, pages in files_to_pages.items():
                try:
                    print(f"   📖 Opening {os.path.basename(file_path)}...")
                    doc = fitz.open(file_path)
                    total_pages = len(doc)
                    file_name = os.path.basename(file_path)
                    
                    # Debug document info
                    print(f"     Document info: {total_pages} pages, encrypted: {doc.is_encrypted}, needs_pass: {doc.needs_pass}")
                    
                    if file_name not in added_pages:
                        added_pages[file_name] = []
                    
                    # Sort pages and filter valid ones (convert to 0-indexed for fitz)
                    valid_pages = []
                    for p in sorted(pages):
                        if 1 <= p <= total_pages:  # p is 1-indexed from metadata
                            valid_pages.append(p - 1)  # Convert to 0-indexed for fitz
                    
                    print(f"     Pages to extract (1-indexed): {[p+1 for p in valid_pages]} (from {total_pages} total)")
                    print(f"     Pages for fitz (0-indexed): {valid_pages}")
                    
                    for i, page_num in enumerate(valid_pages):
                        total_pages_attempted += 1
                        try:
                            # Double-check bounds
                            if page_num < 0 or page_num >= total_pages:
                                print(f"     ⚠️ Skipping page {page_num+1} - out of bounds (valid range: 1-{total_pages})")
                                failed_pages += 1
                                continue
                            
                            # Try to insert the page
                            output_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            
                            # Record original page number for citation (1-indexed for user)
                            added_pages[file_name].append(page_num + 1)
                            total_pages_added += 1
                            
                        except Exception as page_error:
                            print(f"     ❌ Error processing page {page_num+1}: {page_error}")
                            failed_pages += 1
                            continue
                    
                    doc.close()
                    print(f"   ✅ Processed {file_name}: attempted {len(valid_pages)} pages, successfully added {len([p for p in added_pages[file_name]])} pages")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    print(f"   ❌ Error processing {file_path}: {e}")
                    # Count all pages from this file as failed
                    try:
                        if os.path.exists(file_path):
                            doc_for_count = fitz.open(file_path)
                            total_file_pages = len(doc_for_count)
                            doc_for_count.close()
                            file_pages = len([p for p in pages if 1 <= p <= total_file_pages])
                        else:
                            file_pages = 0
                    except:
                        file_pages = 0
                    failed_pages += file_pages
                    total_pages_attempted += file_pages
            
            # Calculate success rate
            success_rate = (total_pages_added / total_pages_attempted) if total_pages_attempted > 0 else 0
            
            print(f"   📊 PDF Reconstruction Summary:")
            print(f"       Pages attempted: {total_pages_attempted}")
            print(f"       Pages successful: {total_pages_added}")
            print(f"       Pages failed: {failed_pages}")
            print(f"       Success rate: {success_rate:.1%}")
            
            # Decision logic: when to use PDF vs fall back to text
            if total_pages_added == 0:
                output_pdf.close()
                print(f"   ❌ No pages could be extracted - falling back to text chunks")
                return self._query_with_chunks(question, chunks)
            elif success_rate < 0.5:  # Less than 50% success
                output_pdf.close()
                print(f"   ⚠️ Low success rate ({success_rate:.1%}) - falling back to text chunks for reliability")
                return self._query_with_chunks(question, chunks)
            elif success_rate < 0.8:  # 50-80% success
                print(f"   ⚠️ Partial success ({success_rate:.1%}) - proceeding with PDF but results may be incomplete")
            else:  # 80%+ success
                print(f"   ✅ High success rate ({success_rate:.1%}) - proceeding with PDF reconstruction")
            
            # Continue with PDF if we have reasonable success
            
            print(f"   📄 Total pages in reconstructed PDF: {total_pages_added}")
            
            if output_pdf.page_count == 0:
                output_pdf.close()
                print(f"   ⚠️ No pages added to PDF, falling back to text chunks")
                return self._query_with_chunks(question, chunks)
            
            # Save temporary PDF and query Gemini
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"temp_slice_{int(time.time())}.pdf")
            output_pdf.save(temp_pdf_path)
            output_pdf.close()
            
            file_size_mb = os.path.getsize(temp_pdf_path) / (1024 * 1024)
            print(f"   📤 Querying Gemini with reconstructed PDF ({total_pages_added} pages, {file_size_mb:.2f} MB)...")
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            with open(temp_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            print_llm_interaction("PDF Slice Query", question, "", truncate=False)
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    question
                ]
            )
            
            result = response.text
            
            print_llm_interaction("PDF Slice Query", "", result, truncate=True)
            
            # Create page range citations
            citations = []
            for file_name, pages in added_pages.items():
                if not pages:
                    continue
                    
                pages = sorted(list(set(pages)))
                ranges = []
                
                if len(pages) > 0:
                    range_start = pages[0]
                    prev_page = pages[0]
                    
                    for page in pages[1:]:
                        if page > prev_page + 1:
                            if range_start == prev_page:
                                ranges.append(f"{range_start}")
                            else:
                                ranges.append(f"{range_start}-{prev_page}")
                            range_start = page
                        prev_page = page
                    
                    # Add the last range
                    if range_start == prev_page:
                        ranges.append(f"{range_start}")
                    else:
                        ranges.append(f"{range_start}-{prev_page}")
                
                citations.append(f"pp. {', '.join(ranges)} ({file_name})")
            
            if citations:
                result += f"\n\nSources: {'; '.join(citations)}"
                print(f"   📋 Added citations: {'; '.join(citations)}")
            
            # Clean up
            try:
                os.remove(temp_pdf_path)
                print(f"   🗑️ Cleaned up temporary PDF")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary PDF: {cleanup_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in PDF slice query: {e}")
            print(f"   ❌ PDF slice error: {e}")
            print(f"   ⚠️ Falling back to text chunks")
            return self._query_with_chunks(question, chunks)
    
    def _query_with_chunks(self, question: str, chunks: List[Document]) -> str:
        """Query with text chunks"""
        
        print(f"   📝 Querying with {len(chunks)} text chunks...")
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk.metadata.get('source', 'Unknown')}, Page: {chunk.metadata.get('page', 'Unknown')}"
            chunk_text = chunk.page_content[:1500]
            context_parts.append(f"[Chunk {i+1}] {source_info}\n{chunk_text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""
        You are analyzing corporate governance documents to answer a specific question.
        
        QUESTION: {question}
        
        DOCUMENT EXCERPTS:
        {context}
        
        Instructions:
        1. Answer the question based ONLY on the provided document excerpts
        2. ALWAYS include specific source citations (page numbers and document names)
        3. If information is not available in the excerpts, state this clearly
        4. Be precise and factual
        5. Format citations as: "According to [document name], page [number]..."
        
        Provide a comprehensive answer with proper source citations.
        """
        
        print_llm_interaction("Text Chunk Query", prompt, "", truncate=True)
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                result = response.text
            else:
                result = self.llm.invoke(prompt)
            
            print_llm_interaction("Text Chunk Query", "", result, truncate=True)
            return result
                
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return f"Error querying documents: {str(e)}"
    
    def _extract_sources_from_chunks(self, chunks: List[Document]) -> List[str]:
        """Extract source information from chunks"""
        sources = []
        for chunk in chunks:
            metadata = chunk.metadata
            source_info = f"Page {metadata.get('page', 'Unknown')} ({metadata.get('source', 'Unknown')})"
            if source_info not in sources:
                sources.append(source_info)
        return sources
    
    def _has_source_citations(self, answer_text: str) -> bool:
        """Check if answer contains source citations"""
        citation_patterns = ['page', 'source:', 'according to', 'document', 'pp.', 'from']
        answer_lower = answer_text.lower()
        return any(pattern in answer_lower for pattern in citation_patterns)
    
    def _assess_confidence(self, answer_text: str, chunks: List[Document]) -> str:
        """Assess confidence based on answer quality and source availability"""
        if len(chunks) >= 5 and len(answer_text) > 200 and self._has_source_citations(answer_text):
            return "high"
        elif len(chunks) >= 3 and len(answer_text) > 100:
            return "medium"
        else:
            return "low"

class OutputGuardrailAgent:
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
    
    def validate_answer(self, answer: Answer) -> Dict[str, Any]:
        """Validate answer quality and source citations"""
        
        # print_agent_action("OUTPUT GUARDRAIL", "Validating Answer Quality", 
        #                   f"Answer length: {len(answer.answer)} chars")
        
        print_agent_action("OUTPUT GUARDRAIL", "Validating Answer Quality", 
                  f"Answer length: {len(answer.answer)} chars", "Rule-based validation")
        
        validation_result = {
            "has_answer": len(answer.answer.strip()) > 20,
            "has_citations": answer.has_citations,
            "confidence": answer.confidence,
            "sources_count": len(answer.sources),
            "retrieval_method": self.config.retrieval_method,
            "issues": [],
            "approved": False
        }
        
        print(f"   📏 Answer length: {len(answer.answer)} characters")
        print(f"   📚 Sources: {validation_result['sources_count']}")
        print(f"   📋 Has citations: {validation_result['has_citations']}")
        print(f"   🎯 Confidence: {validation_result['confidence']}")
        
        # Check for substantive answer
        if not validation_result["has_answer"]:
            validation_result["issues"].append("Answer is too short or empty")
            print(f"   ❌ Answer too short")
        
        # Check for citations
        if not validation_result["has_citations"]:
            validation_result["issues"].append("Answer lacks source citations")
            print(f"   ❌ Missing citations")
        
        # Check confidence
        if answer.confidence == "low":
            validation_result["issues"].append("Low confidence in answer quality")
            print(f"   ⚠️ Low confidence")
        
        # Validation based on retrieval method
        if self.config.retrieval_method == "direct":
            validation_result["approved"] = (
                validation_result["has_answer"] and 
                validation_result["has_citations"]
            )
        elif self.config.retrieval_method in ["hybrid", "bm25", "vector"]:
            if validation_result["sources_count"] < 2:
                validation_result["issues"].append(f"Limited source coverage for {self.config.retrieval_method} retrieval")
                print(f"   ⚠️ Limited source coverage")
            
            validation_result["approved"] = (
                validation_result["has_answer"] and 
                validation_result["has_citations"] and 
                validation_result["sources_count"] >= 1 and
                len(validation_result["issues"]) <= 1
            )
        
        status = "✅ APPROVED" if validation_result["approved"] else "❌ REJECTED"
        print(f"   {status} - Issues: {len(validation_result['issues'])}")
        
        if validation_result["issues"]:
            for issue in validation_result["issues"]:
                print(f"     • {issue}")
        
        return validation_result

# class ScoringAgent:
    
#     def __init__(self, config: OptimizedConfig):
#         self.config = config
#         self.llm_manager = LLMManager(config)
#         self.llm = None
#         self.current_model = None
#         self._setup_llm()
    
#     def _setup_llm(self):
#         """Setup LLM for scoring"""
#         self.llm, self.current_model = self.llm_manager.get_llm("scoring_agent")
        
#     def score_topic(self, topic: TopicDefinition, answers: List[Answer]) -> Dict[str, Any]:
#         """Score the topic based on collected answers and rubric"""
        
#         # print_agent_action("SCORING", "Final Topic Scoring", 
#         #                   f"Evaluating {len(answers)} pieces of evidence")
        
#         print_agent_action("SCORING", "Final Topic Scoring", 
#                   f"Evaluating {len(answers)} pieces of evidence", self.current_model)
        
#         if not self.llm:
#             print(f"   ❌ No LLM available for scoring")
#             return {
#                 "score": 0,
#                 "justification": "No LLM available for scoring",
#                 "confidence": "low",
#                 "evidence_quality": "poor",
#                 "retrieval_method": self.config.retrieval_method
#             }
        
#         # Prepare evidence summary
#         evidence_summary = self._prepare_evidence_summary(answers)
#         print(f"   📊 Evidence summary prepared ({len(evidence_summary)} characters)")
        
#         prompt = f"""
#         You are scoring a corporate governance topic based on collected research evidence.
        
#         TOPIC: {topic.topic_name}
#         GOAL: {topic.goal}
#         GUIDANCE: {topic.guidance}
        
#         SCORING RUBRIC:
#         {json.dumps(topic.scoring_rubric, indent=2)}
        
#         RESEARCH EVIDENCE (collected using {self.config.retrieval_method} retrieval):
#         {evidence_summary}
        
#         Instructions:
#         1. Evaluate the evidence against each scoring level in the rubric
#         2. Assign a score (0, 1, or 2) based on which level best matches the evidence
#         3. Provide detailed justification with specific references to the evidence
#         4. Preserve all source citations from the evidence in your justification
        
#         Respond in JSON format:
#         {{
#             "score": 0/1/2,
#             "justification": "Detailed justification with source citations",
#             "evidence_quality": "excellent/good/fair/poor",
#             "key_findings": ["list of key findings that influenced the score"]
#         }}
        
#         Be objective and base your score strictly on the evidence provided.
#         """
        
#         try:
#             if hasattr(self.llm, 'generate_content'):
#                 response = self.llm.generate_content(prompt)
#                 response_text = response.text
#             else:
#                 response_text = self.llm.invoke(prompt)
            
#             # Parse response
#             import re
#             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
                
#                 # Validate score
#                 score = result.get("score", 0)
#                 if score not in [0, 1, 2]:
#                     score = 0
#                     print(f"   ⚠️ Invalid score, defaulting to 0")
                
#                 confidence = self._assess_scoring_confidence(answers)
                
#                 print(f"   📊 Final Score: {score}/2")
#                 print(f"   🎯 Evidence Quality: {result.get('evidence_quality', 'fair')}")
#                 print(f"   💪 Scoring Confidence: {confidence}")
                
#                 return {
#                     "score": score,
#                     "justification": result.get("justification", "No justification provided"),
#                     "evidence_quality": result.get("evidence_quality", "fair"),
#                     "key_findings": result.get("key_findings", []),
#                     "confidence": confidence,
#                     "retrieval_method": self.config.retrieval_method
#                 }
#             else:
#                 raise ValueError("No JSON found in response")
                
#         except Exception as e:
#             logger.error(f"Scoring error: {e}")
#             print(f"   ❌ Scoring failed: {str(e)}")
#             return {
#                 "score": 0,
#                 "justification": f"Scoring failed: {str(e)}",
#                 "confidence": "low",
#                 "evidence_quality": "poor",
#                 "retrieval_method": self.config.retrieval_method
#             }
    
#     def _prepare_evidence_summary(self, answers: List[Answer]) -> str:
#         """Prepare a summary of all research evidence"""
#         summary_parts = []
        
#         for i, answer in enumerate(answers, 1):
#             part = f"""
# EVIDENCE {i}:
# Question: {answer.question}
# Answer: {answer.answer}
# Sources: {', '.join(answer.sources)}
# Confidence: {answer.confidence}
# ---
# """
#             summary_parts.append(part)
        
#         return "\n".join(summary_parts)
    
#     def _assess_scoring_confidence(self, answers: List[Answer]) -> str:
#         """Assess overall confidence in scoring based on answer quality"""
#         if not answers:
#             return "low"
        
#         high_confidence_count = sum(1 for ans in answers if ans.confidence == "high" and ans.has_citations)
#         total_answers = len(answers)
        
#         if high_confidence_count >= total_answers * 0.8:
#             return "high"
#         elif high_confidence_count >= total_answers * 0.5:
#             return "medium"
#         else:
#             return "low"

class ScoringAgent:
    r"""
    ScoringAgent Class - Final Evaluation and Scoring System (COMPLETE REWRITE)
    ============================================================================

    FIXED: Eliminates evidence numbers entirely. Creates merged evidence summary with direct citations.
    The LLM will only see actual document content and source lists, forcing proper citation behavior.

    WHY THIS APPROACH:
    -----------------
    - No "Evidence 1", "Evidence 2" references possible
    - Forces LLM to cite actual page numbers and document names
    - Creates coherent narrative from all research findings
    - Maintains all source citations in easily accessible format
    - Provides cleaner, more professional justifications

    EXAMPLE OUTPUT TRANSFORMATION:
    Before: "Evidence 1 states that women comprised 11% of workforce (page 12 of the report)"
    After:  "According to page 12 of workforce_diversity_report.pdf, women comprised 11% of the workforce"
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM for scoring"""
        self.llm, self.current_model = self.llm_manager.get_llm("scoring_agent")
        
    def score_topic(self, topic: TopicDefinition, answers: List[Answer]) -> Dict[str, Any]:
        """Score the topic based on collected answers and rubric"""
        
        print_agent_action("SCORING", "Final Topic Scoring", 
                  f"Evaluating {len(answers)} pieces of evidence", self.current_model)
        
        if not self.llm:
            print(f"   ❌ No LLM available for scoring")
            return {
                "score": 0,
                "justification": "No LLM available for scoring",
                "confidence": "low",
                "evidence_quality": "poor",
                "retrieval_method": self.config.retrieval_method
            }
        
        # Create merged evidence summary (NO evidence numbers)
        evidence_summary = self._create_merged_evidence_summary(answers)
        print(f"   📊 Merged evidence summary prepared ({len(evidence_summary)} characters)")
        
        prompt = f"""
You are scoring a corporate governance topic based on research findings from corporate documents.

TOPIC: {topic.topic_name}
GOAL: {topic.goal}
GUIDANCE: {topic.guidance}

SCORING RUBRIC:
{json.dumps(topic.scoring_rubric, indent=2)}

{evidence_summary}

SCORING INSTRUCTIONS:
1. Analyze all the research findings above against the scoring rubric
2. Determine which rubric level (0, 1, or 2) best matches the findings
3. Provide detailed justification using ONLY the specific page numbers and document names listed in the "SOURCES REFERENCED" section
4. Your citations must be in format: "According to page X of document.pdf..." or "As stated on page Y of report.pdf..."
5. Do not make up or assume any information not present in the research findings

RESPONSE FORMAT (JSON):
{{
    "score": 0/1/2,
    "justification": "Detailed explanation with specific page and document citations from the sources listed above",
    "evidence_quality": "excellent/good/fair/poor",
    "key_findings": ["List of specific findings that influenced the score"]
}}

Base your score strictly on the research findings and cite sources using the exact page numbers and document names provided.
"""
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self.llm.invoke(prompt)
            
            # Parse response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate score
                score = result.get("score", 0)
                if score not in [0, 1, 2]:
                    score = 0
                    print(f"   ⚠️ Invalid score, defaulting to 0")
                
                confidence = self._assess_scoring_confidence(answers)
                
                print(f"   📊 Final Score: {score}/2")
                print(f"   🎯 Evidence Quality: {result.get('evidence_quality', 'fair')}")
                print(f"   💪 Scoring Confidence: {confidence}")
                
                return {
                    "score": score,
                    "justification": result.get("justification", "No justification provided"),
                    "evidence_quality": result.get("evidence_quality", "fair"),
                    "key_findings": result.get("key_findings", []),
                    "confidence": confidence,
                    "retrieval_method": self.config.retrieval_method
                }
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            print(f"   ❌ Scoring failed: {str(e)}")
            return {
                "score": 0,
                "justification": f"Scoring failed: {str(e)}",
                "confidence": "low",
                "evidence_quality": "poor",
                "retrieval_method": self.config.retrieval_method
            }
    
    def _create_merged_evidence_summary(self, answers: List[Answer]) -> str:
        """
        Create a comprehensive evidence summary that eliminates evidence numbers entirely.
        Merges all findings into coherent research content with complete source tracking.
        """
        if not answers:
            return """
RESEARCH FINDINGS:
No research findings were collected.

SOURCES REFERENCED:
None
"""
        
        # Step 1: Extract and merge all substantive content
        all_findings = []
        all_sources = set()
        question_topics = []
        
        for answer in answers:
            # Collect the actual research content
            if answer.answer and len(answer.answer.strip()) > 20:
                # Clean up the answer text
                finding = answer.answer.strip()
                
                # Add context about what was being researched (without evidence numbers)
                if answer.question:
                    question_topics.append(f"Research on: {answer.question}")
                
                all_findings.append(finding)
            
            # Collect all unique sources
            for source in answer.sources:
                if source and source.strip():
                    all_sources.add(source.strip())
        
        # Step 2: Create merged findings section
        if all_findings:
            # Combine findings with natural separators
            merged_content = "\n\n".join([
                f"Research Finding: {finding}" for finding in all_findings
            ])
        else:
            merged_content = "No substantive research findings were collected."
        
        # Step 3: Create comprehensive sources section
        sources_section = self._create_sources_section(all_sources)
        
        # Step 4: Create research context section
        context_section = ""
        if question_topics:
            unique_topics = list(set(question_topics))
            context_section = f"""
RESEARCH AREAS INVESTIGATED:
{chr(10).join([f"• {topic}" for topic in unique_topics])}
"""
        
        # Step 5: Assemble final summary
        summary = f"""
COMPREHENSIVE RESEARCH FINDINGS:
{merged_content}

{context_section}
{sources_section}

CITATION REQUIREMENTS:
When writing your justification, you must cite information using the specific page numbers and document names listed in "SOURCES REFERENCED" above. Use format: "According to page X of document_name.pdf..." or "As stated in page Y of report_name.pdf..."
"""
        
        return summary
    
    def _create_sources_section(self, sources: set) -> str:
        """
        Create a well-organized sources section for easy citation by the LLM.
        Groups sources by document and formats them clearly.
        """
        if not sources:
            return """
SOURCES REFERENCED:
No sources were identified in the research.
"""
        
        # Group sources by document
        doc_sources = {}
        other_sources = []
        
        for source in sorted(sources):
            # Parse sources in format: "Page 15 (governance_report.pdf)"
            if "(" in source and ")" in source and source.startswith("Page "):
                try:
                    # Extract page and document
                    page_part = source.split("(")[0].strip()  # "Page 15"
                    doc_part = source.split("(")[1].replace(")", "").strip()  # "governance_report.pdf"
                    
                    if doc_part not in doc_sources:
                        doc_sources[doc_part] = []
                    doc_sources[doc_part].append(page_part)
                except:
                    other_sources.append(source)
            else:
                other_sources.append(source)
        
        # Build sources section
        sources_lines = ["SOURCES REFERENCED:"]
        
        # Add document-grouped sources
        for doc_name, pages in sorted(doc_sources.items()):
            if len(pages) == 1:
                sources_lines.append(f"• {pages[0]} of {doc_name}")
            else:
                # Sort pages numerically if possible
                try:
                    page_numbers = []
                    for page in pages:
                        if page.startswith("Page "):
                            page_numbers.append(int(page.replace("Page ", "")))
                    page_numbers.sort()
                    formatted_pages = [f"Page {num}" for num in page_numbers]
                    sources_lines.append(f"• {', '.join(formatted_pages)} of {doc_name}")
                except:
                    sources_lines.append(f"• {', '.join(sorted(pages))} of {doc_name}")
        
        # Add other sources
        for source in sorted(other_sources):
            sources_lines.append(f"• {source}")
        
        return "\n".join(sources_lines)
    
    def _assess_scoring_confidence(self, answers: List[Answer]) -> str:
        """Assess overall confidence in scoring based on answer quality"""
        if not answers:
            return "low"
        
        high_confidence_count = sum(1 for ans in answers if ans.confidence == "high" and ans.has_citations)
        total_answers = len(answers)
        
        if high_confidence_count >= total_answers * 0.8:
            return "high"
        elif high_confidence_count >= total_answers * 0.5:
            return "medium"
        else:
            return "low"
    
    def _prepare_evidence_summary(self, answers: List[Answer]) -> str:
        """
        DEPRECATED: Replaced with _create_merged_evidence_summary
        This method is kept for compatibility but redirects to the new approach
        """
        return self._create_merged_evidence_summary(answers)

class OptimizedAgenticOrchestrator:
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        
        # Initialize agents
        self.input_guardrail = InputGuardrailAgent(config)
        self.question_agent = QuestionAgent(config)
        self.research_agent = OptimizedResearchAgent(config)  # Optimized version
        self.output_guardrail = OutputGuardrailAgent(config)
        self.scoring_agent = ScoringAgent(config)
        # Display agent LLM configuration
        self._display_agent_configuration()
        
        # State tracking
        self.current_topic = None
        self.answers = []
        self.max_iterations = 3
        
        print_section("OPTIMIZED AGENTIC ORCHESTRATOR READY", 
                     f"Retrieval: {config.retrieval_method}, Max Iterations: {self.max_iterations}\n" +
                     f"✅ All chunks pre-computed and cached\n" +
                     f"✅ All retrievers pre-built\n" +
                     f"⚡ Ready for FAST queries", Colors.OKGREEN)
    
    def evaluate_topic(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Main evaluation workflow with optimized retrieval"""
        
        print_section("STARTING OPTIMIZED TOPIC EVALUATION", 
                     f"Topic: {topic.topic_name}\nMethod: {self.config.retrieval_method}", 
                     Colors.HEADER)
        
        total_start_time = time.time()
        
        # Step 1: Validate input
        print_section("STEP 1: INPUT VALIDATION")
        input_validation = self.input_guardrail.validate_topic_definition(topic)
        if not input_validation.get("valid", True):
            print(f"   ❌ Validation failed")
            return {
                "success": False,
                "error": "Invalid topic definition",
                "issues": input_validation.get("issues", []),
                "suggestions": input_validation.get("suggestions", [])
            }
        
        # Initialize state
        self.current_topic = topic
        self.answers = []
        iteration = 0
        
        # Step 2: Generate initial question
        print_section("STEP 2: INITIAL QUESTION GENERATION")
        question_start = time.time()
        current_question = self.question_agent.generate_initial_question(topic)
        question_time = time.time() - question_start
        print(f"   ⏱️ Question generation: {question_time:.3f}s")
        
        # Step 3: Iterative research with OPTIMIZED retrieval
        print_section("STEP 3: OPTIMIZED ITERATIVE RESEARCH")
        
        total_research_time = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print_section(f"RESEARCH ITERATION {iteration}", 
                         f"Question: {current_question.text}", 
                         Colors.WARNING)
            
            # FAST research using pre-computed data
            research_start = time.time()
            answer = self.research_agent.research_question(current_question.text)
            research_time = time.time() - research_start
            total_research_time += research_time
            
            print(f"   ⚡ Research iteration completed in {research_time:.3f}s")
            
            # Validate the answer
            validation = self.output_guardrail.validate_answer(answer)
            
            if validation["approved"]:
                self.answers.append(answer)
                print(f"   ✅ Answer approved and added to evidence pool")
            else:
                print(f"   ⚠️ Answer has issues but still added to evidence pool")
                self.answers.append(answer)
            
            print(f"   📊 Current evidence count: {len(self.answers)}")
            
            # Determine if we need more information
            follow_up_question = self.question_agent.generate_follow_up_question(topic, self.answers)
            
            if follow_up_question is None:
                print(f"   ✅ No more questions needed, ready for scoring")
                break
            else:
                print(f"   ➡️ Follow-up question generated")
                current_question = follow_up_question
        
        # Step 4: Final scoring
        print_section("STEP 4: FINAL SCORING")
        scoring_start = time.time()
        scoring_result = self.scoring_agent.score_topic(topic, self.answers)
        scoring_time = time.time() - scoring_start
        print(f"   ⏱️ Scoring completed in {scoring_time:.3f}s")
        
        # Step 5: Compile final result
        print_section("STEP 5: COMPILING RESULTS")
        
        total_time = time.time() - total_start_time
        
        result = {
            "success": True,
            "topic": {
                "name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "rubric": topic.scoring_rubric
            },
            "research_summary": {
                "iterations": iteration,
                "questions_asked": len(self.answers),
                "answers_approved": len([a for a in self.answers if self.output_guardrail.validate_answer(a)["approved"]]),
                "retrieval_method": self.config.retrieval_method,
                "total_sources": len(set([s for ans in self.answers for s in ans.sources])),
                "pdf_slices_used": self.config.use_pdf_slices,
                "optimization_enabled": True,
                "agent_llm_config": {
                    "input_agent": getattr(self.input_guardrail, 'current_model', 'Unknown'),
                    "question_agent": getattr(self.question_agent, 'current_model', 'Unknown'),
                    "research_agent": getattr(self.research_agent, 'current_model', 'Unknown'),
                    "scoring_agent": getattr(self.scoring_agent, 'current_model', 'Unknown')
                }
            },
            "performance_metrics": {
                "total_time": total_time,
                "research_time": total_research_time,
                "scoring_time": scoring_time,
                "avg_research_per_iteration": total_research_time / iteration if iteration > 0 else 0
            },
            "evidence": [
                {
                    "question": ans.question,
                    "answer": ans.answer,
                    "sources": ans.sources,
                    "confidence": ans.confidence,
                    "has_citations": ans.has_citations
                }
                for ans in self.answers
            ],
            "scoring": scoring_result,
            "timestamp": datetime.now().isoformat()
        }
        
        print_section("OPTIMIZED EVALUATION COMPLETED", 
                     f"Final Score: {scoring_result.get('score', 'N/A')}/2\n" +
                     f"Total Time: {total_time:.3f}s (Research: {total_research_time:.3f}s)\n" +
                     f"Avg Research/Iteration: {result['performance_metrics']['avg_research_per_iteration']:.3f}s\n" +
                     f"Sources Used: {result['research_summary']['total_sources']}\n" +
                     f"Method: {self.config.retrieval_method} (Optimized)",
                     Colors.OKGREEN)
        
        print("check_done")
        return result

    def _display_agent_configuration(self):
        """Display current agent LLM configuration"""
        print_section("AGENT LLM CONFIGURATION", color=Colors.HEADER)
        
        # Get actual models being used
        input_model = getattr(self.input_guardrail, 'current_model', 'Unknown')
        question_model = getattr(self.question_agent, 'current_model', 'Unknown')
        research_model = getattr(self.research_agent, 'current_model', 'Unknown')
        scoring_model = getattr(self.scoring_agent, 'current_model', 'Unknown')
        
        print(f"   🛡️  Input Guardrail Agent:  {input_model}")
        print(f"   ❓  Question Agent:         {question_model}")
        print(f"   🔍  Research Agent:         {research_model}")
        print(f"   🏛️  Output Guardrail Agent: Rule-based (no LLM)")
        print(f"   📊  Scoring Agent:          {scoring_model}")
        
        # Show model distribution
        models_used = [input_model, question_model, research_model, scoring_model]
        gemini_count = len([m for m in models_used if m.startswith('gemini')])
        ollama_count = len([m for m in models_used if not m.startswith('gemini') and m != 'Unknown'])
        
        print(f"\n   📈 Model Distribution:")
        if gemini_count > 0:
            print(f"      🌩️  Gemini models: {gemini_count}/4 agents")
        if ollama_count > 0:
            print(f"      🏠  Local models:  {ollama_count}/4 agents")
        
        print()
    
def load_sample_topics():
    """Load sample topics from JSON file"""
    try:
        # Try to load from current directory first
        json_path = "sample_topics.json"
        if not os.path.exists(json_path):
            # Try relative path if not in current directory
            json_path = os.path.join(os.path.dirname(__file__), "sample_topics.json")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert string keys to integers for easier access
        topics_data = {}
        for key, value in data["topics"].items():
            topics_data[int(key)] = value
            
        return topics_data
        
    except FileNotFoundError:
        print("⚠️ sample_topics.json file not found. Using fallback topics.")
        return get_fallback_topics()
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing sample_topics.json: {e}. Using fallback topics.")
        return get_fallback_topics()
    except Exception as e:
        print(f"⚠️ Unexpected error loading topics: {e}. Using fallback topics.")
        return get_fallback_topics()

def get_fallback_topics():
    """Fallback topics in case JSON file is not available"""
    return {
        1: {
            "topic_name": "Board Independence",
            "goal": "To assess if the board have directors with permanent board seats",
            "guidance": "Basic evaluation of board member tenure and independence status.",
            "scoring_rubric": {
                "0": "Poor board independence",
                "1": "Moderate board independence", 
                "2": "Excellent board independence"
            }
        }
    }

# def create_sample_topic() -> TopicDefinition:
#     """Create a sample topic for testing"""
#     return TopicDefinition(
#         topic_name="Board Independence",
#         goal="To assess if the board have directors with permanent board seats",
#         guidance="You need to look for the corporate governance report. Find the reappointment date for each board members. If the reppointment date is either not provided or older than 5 years (i.e some date before 2019), then you need to check appointment date. If appointment date is also older than 5 years (i.e before 2019), mark that board member as permanent. Give list of board members and whether or not they are permanent. In other words, either of appointment date or reappointment date should be within last 5 years. For example, if a board member has appoinment date '02-07-2020' and reappointment date is not present, then because the appointment date is within last 5 years  (i.e March 2020 to March 2025 assuming we are checking for annual report as of 31st March 2025) then we would label them as 'Not permanent'. Second example, if any board member has appointment date as 01-01-2012 and reappointment date not present, then we would mark them permanent. Do not present output in table format. Give me text based paragraphs. You are looking at the corporate governance report as of 31st March 2024. Make sure you quote this source in the answer with the page number from which you extract the information ",
#         scoring_rubric={
#             "0": "if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders.",
#             "1": "if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so.",
#             "2": "if All directors are marked as non-permanent board members"
#         }
#     )

# def create_sample_topic(topic_number: int = 1) -> TopicDefinition:
#     """Create a sample topic for testing based on topic number (1-5)"""
    
#     # Predefined topics mapping
#     topics_data = {
#         1: {
#             "topic_name": "Board Independence",
#             "goal": "To assess if the board have directors with permanent board seats",
#             "guidance": "You need to look for the corporate governance report. Find the reappointment date for each board members. If the reappointment date is either not provided or older than 5 years (i.e some date before 2019), then you need to check appointment date. If appointment date is also older than 5 years (i.e before 2019), mark that board member as permanent. Give list of board members and whether or not they are permanent. In other words, either of appointment date or reappointment date should be within last 5 years. For example, if a board member has appointment date '02-07-2020' and reappointment date is not present, then because the appointment date is within last 5 years (i.e March 2020 to March 2025 assuming we are checking for annual report as of 31st March 2025) then we would label them as 'Not permanent'. Second example, if any board member has appointment date as 01-01-2012 and reappointment date not present, then we would mark them permanent. Do not present output in table format. Give me text based paragraphs. You are looking at the corporate governance report as of 31st March 2024. Make sure you quote this source in the answer with the page number from which you extract the information.",
#             "scoring_rubric": {
#                 "0": "if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders.",
#                 "1": "if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so.",
#                 "2": "if All directors are marked as non-permanent board members"
#             }
#         },
#         2: {
#             "topic_name": "AGM Delay",
#             "goal": "To evaluate if the company's books were audited quick enough that they could hold AGM quicker after the financial year ended.",
#             "guidance": "You have to look for two dates. One for financial year end date and second for AGM date. Financial year end date is usually 31st March, but it can be different for some companies. This you can typically find in annual report. Then you need to check the AGM date. AGM date is typically found on notice section of annual report. It is either on first few pages or last few pages of the annual report. Make sure you quote this source in the answer with the page number from which you extract the information. Calculate the gap between financial year end date and AGM date. And answer whether this gap is less than 4 months, between 4 to 6 months or more than 6 months.",
#             "scoring_rubric": {
#                 "0": "if the gap between financial year end date and AGM date is more than 6 months",
#                 "1": "if the gap between financial year end date and AGM date is between 4 to 6 months",
#                 "2": "if the gap between financial year end date and AGM date is less than 4 months"
#             }
#         },
#         3: {
#             "topic_name": "POSH Compliance",
#             "goal": "To assess if the company has a proper POSH (Prevention of Sexual Harassment) policy and compliance",
#             "guidance": "You have to check two main things. First if the company has publicaly available POSH policy.  Second, check if the company has reported any complaints or cases under this policy in the last financial year. This information is typically found in the corporate governance report or the management discussion and analysis section of the annual report. Make sure you quote this source in the answer with the page number from which you extract the information.",
#             "scoring_rubric": {
#                 "0": "if company does not have any publicly disclosed policy regarding prevention of sexual harrassment and the company also has not provided information on the number of sexual harassment incidents.",
#                 "1": "if company has either publicly disclosed policy regarding prevention of sexual harrassment or the company has provided information on the number of sexual harassment incidents.",
#                 "2": "if company has both publicly disclosed policy regarding prevention of sexual harrassment and the company has provided information on the number of sexual harassment incidents."
#             }
#         },
#         4: {
#             "topic_name": "Related Party Transactions Oversight",
#             "goal": "To evaluate the governance framework for managing related party transactions",
#             "guidance": "Examine the corporate governance policies and annual report disclosures regarding related party transactions. Look for information about the approval process, committee oversight, materiality thresholds, and disclosure practices. Check if there's a clear policy for identifying, evaluating, and approving related party transactions. Verify if the audit committee or independent directors have proper oversight role. Also assess the quality of disclosures about actual related party transactions during the year. Make sure to provide specific page citations.",
#             "scoring_rubric": {
#                 "0": "if there are inadequate policies for related party transaction oversight or poor disclosure practices",
#                 "1": "if there are basic policies and oversight mechanisms but with some gaps in process or disclosure quality",
#                 "2": "if there are comprehensive policies, strong oversight by independent committees, and transparent disclosures"
#             }
#         },
#         5: {
#             "topic_name": "Women representation in workforce",
#             "goal": "To assess if company has sufficient women representation in workforce",
#             "guidance": "You need to look for percentage of women in total workforce. This information is typically found in the corporate governance report or the annual report. If direct ratio is not given, try to look for total women employees and total number of employees and calculate ratio yourself. Make sure you quote this source in the answer with the page number from which you extract the information.",
#             "scoring_rubric": {                                                 
#                 "0": "if there is no such disclosure or the percentage of women in workforce is less than 10%",
#                 "1": "if percentage of  women in workforce is between 10% to 30%",
#                 "2": "if percentage of  women in workforce is more than 30%"
#             }
#         }
#     }
    
#     # Validate topic number
#     if topic_number not in topics_data:
#         print(f"⚠️ Invalid topic number {topic_number}. Available topics: 1-5. Defaulting to topic 1.")
#         topic_number = 1
    
#     # Get the selected topic data
#     topic_data = topics_data[topic_number]
    
#     print(f"📋 Selected Topic {topic_number}: {topic_data['topic_name']}")
    
#     return TopicDefinition(
#         topic_name=topic_data["topic_name"],
#         goal=topic_data["goal"],
#         guidance=topic_data["guidance"],
#         scoring_rubric=topic_data["scoring_rubric"]
#     )

def create_sample_topic(topic_number: int = 1) -> TopicDefinition:
    """Create a sample topic for testing based on topic number (1-5)"""
    
    # Load topics from JSON file
    topics_data = load_sample_topics()
    
    # Validate topic number
    if topic_number not in topics_data:
        available_topics = list(topics_data.keys())
        print(f"⚠️ Invalid topic number {topic_number}. Available topics: {available_topics}. Defaulting to topic 1.")
        topic_number = 1
    
    # Get the selected topic data
    topic_data = topics_data[topic_number]
    
    print(f"📋 Selected Topic {topic_number}: {topic_data['topic_name']}")
    
    return TopicDefinition(
        topic_name=topic_data["topic_name"],
        goal=topic_data["goal"],
        guidance=topic_data["guidance"],
        scoring_rubric=topic_data["scoring_rubric"]
    )

def list_available_topics() -> None:
    """Display all available sample topics"""
    try:
        topics_data = load_sample_topics()
        
        print("📚 Available Sample Topics:")
        for num, topic_data in topics_data.items():
            print(f"   {num}. {topic_data['topic_name']}")
        print()
        
    except Exception as e:
        print(f"⚠️ Error loading topics for display: {e}")
        print("📚 Available Sample Topics:")
        print("   1. Board Independence (fallback)")
        print()

def get_user_topic_choice() -> int:
    """Get topic choice from user with validation"""
    print_section("TOPIC SELECTION", color=Colors.OKBLUE)
    
    # Show available topics
    list_available_topics()
    
    # Get available topic numbers
    topics_data = load_sample_topics()
    available_numbers = list(topics_data.keys())
    
    while True:
        try:
            choice = input(f"🔍 Please select a topic number ({min(available_numbers)}-{max(available_numbers)}): ").strip()
            
            if not choice:
                print("⚠️ Please enter a topic number.")
                continue
                
            topic_num = int(choice)
            
            if topic_num in available_numbers:
                return topic_num
            else:
                print(f"⚠️ Please enter a number from: {available_numbers}")
                
        except ValueError:
            print("⚠️ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def save_results(result: Dict[str, Any], config: OptimizedConfig):
    """Save evaluation results to file"""
    try:
        print_section("SAVING RESULTS")
        
        # Create results directory
        results_dir = os.path.join(config.base_path, "96_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_name_clean = result['topic']['name'].replace(' ', '_').replace('/', '_')
        filename = f"optimized_evaluation_{topic_name_clean}_{config.retrieval_method}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"   💾 Detailed results saved: {filepath}")
        
        # Also save summary CSV
        save_summary_csv(result, config)
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def save_summary_csv(result: Dict[str, Any], config: OptimizedConfig):
    """Save summary to CSV for easy analysis"""
    try:
        import pandas as pd
        
        # Create summary data with performance metrics
        summary_data = {
            'timestamp': [result['timestamp']],
            'company': [config.company],
            'topic_name': [result['topic']['name']],
            'final_score': [result['scoring']['score']],
            'evidence_quality': [result['scoring']['evidence_quality']],
            'confidence': [result['scoring']['confidence']],
            'questions_asked': [result['research_summary']['questions_asked']],
            'answers_approved': [result['research_summary']['answers_approved']],
            'iterations': [result['research_summary']['iterations']],
            'retrieval_method': [result['research_summary']['retrieval_method']],
            'unique_sources': [result['research_summary']['total_sources']],
            'pdf_slices_used': [result['research_summary']['pdf_slices_used']],
            'optimization_enabled': [result['research_summary']['optimization_enabled']],
            'total_time': [result['performance_metrics']['total_time']],
            'research_time': [result['performance_metrics']['research_time']],
            'avg_research_per_iteration': [result['performance_metrics']['avg_research_per_iteration']],
            'justification': [result['scoring']['justification']]
        }
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        results_dir = os.path.join(config.base_path, "96_results")
        csv_path = os.path.join(results_dir, "optimized_evaluations_summary.csv")
        
        # Append to existing CSV or create new
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        
        combined_df.to_csv(csv_path, index=False)
        print(f"   📈 Summary CSV updated: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving summary CSV: {e}")

def test_optimization_performance(company: str = "TATAMOTORS"):
    """Test and compare optimized vs non-optimized performance"""
    
    print_section("OPTIMIZATION PERFORMANCE TEST", 
                 f"Company: {company}\nTesting speed improvements", 
                 Colors.HEADER)
    
    topic = create_sample_topic()
    results = {}
    
    # Test optimized version
    print_section("TESTING OPTIMIZED VERSION", color=Colors.OKGREEN)
    config_opt = OptimizedConfig(company)
    config_opt.retrieval_method = "hybrid"
    
    orchestrator_opt = OptimizedAgenticOrchestrator(config_opt)
    
    start_time = time.time()
    result_opt = orchestrator_opt.evaluate_topic(topic)
    end_time = time.time()
    
    if result_opt["success"]:
        results["optimized"] = {
            "total_time": end_time - start_time,
            "research_time": result_opt['performance_metrics']['research_time'],
            "score": result_opt['scoring']['score'],
            "sources": result_opt['research_summary']['total_sources'],
            "iterations": result_opt['research_summary']['iterations']
        }
        
        print(f"   ⚡ Optimized Total Time: {results['optimized']['total_time']:.3f}s")
        print(f"   🔍 Research Time: {results['optimized']['research_time']:.3f}s")
        print(f"   📊 Score: {results['optimized']['score']}/2")
        
        save_results(result_opt, config_opt)
    
    # Print performance summary
    print_section("OPTIMIZATION PERFORMANCE SUMMARY", color=Colors.HEADER)
    
    if "optimized" in results:
        opt = results["optimized"]
        print(f"✅ OPTIMIZED VERSION:")
        print(f"   Total Time: {opt['total_time']:.3f}s")
        print(f"   Research Time: {opt['research_time']:.3f}s")
        print(f"   Avg Research/Iteration: {opt['research_time']/opt['iterations']:.3f}s")
        print(f"   Score: {opt['score']}/2")
        print(f"   Sources: {opt['sources']}")
        
        print(f"\n🚀 KEY OPTIMIZATIONS:")
        print(f"   • Pre-computed chunks (no repeated chunking)")
        print(f"   • Pre-built retrievers (no repeated indexing)")
        print(f"   • Cached embeddings (no repeated vectorization)")
        print(f"   • Fast semantic search with pre-computed vectors")
        print(f"   • Intelligent caching system")

def test_all_retrieval_methods_optimized(company: str = "TATAMOTORS"):
    """Test all retrieval methods with optimization"""
    
    retrieval_methods = ["hybrid", "bm25", "vector", "direct"]
    results = {}
    
    print_section("TESTING ALL OPTIMIZED RETRIEVAL METHODS", 
                 f"Company: {company}\nMethods: {', '.join(retrieval_methods)}", 
                 Colors.HEADER)
    
    for method in retrieval_methods:
        print_section(f"TESTING OPTIMIZED {method.upper()} METHOD", color=Colors.WARNING)
        
        config = OptimizedConfig(company)
        config.retrieval_method = method
        
        orchestrator = OptimizedAgenticOrchestrator(config)
        topic = create_sample_topic()
        
        start_time = time.time()
        result = orchestrator.evaluate_topic(topic)
        end_time = time.time()
        
        if result["success"]:
            results[method] = {
                "score": result['scoring']['score'],
                "confidence": result['scoring']['confidence'],
                "sources": result['research_summary']['total_sources'],
                "time": end_time - start_time,
                "research_time": result['performance_metrics']['research_time'],
                "iterations": result['research_summary']['iterations']
            }
            
            print(f"   ✅ Score: {result['scoring']['score']}/2")
            print(f"   ⚡ Total Time: {end_time - start_time:.3f}s")
            print(f"   🔍 Research Time: {result['performance_metrics']['research_time']:.3f}s")
            print(f"   📚 Sources: {result['research_summary']['total_sources']}")
            
            save_results(result, config)
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Print comparison
    print_section("OPTIMIZED RETRIEVAL METHOD COMPARISON", color=Colors.HEADER)
    
    print(f"{'METHOD':<10} {'SCORE':<8} {'TOTAL':<10} {'RESEARCH':<10} {'SOURCES':<8} {'CONF'}")
    print("-" * 70)
    
    for method, metrics in results.items():
        print(f"{method.upper():<10} {metrics['score']}/2{'':<5} {metrics['time']:.2f}s{'':<5} {metrics['research_time']:.2f}s{'':<5} {metrics['sources']:<8} {metrics.get('confidence', 'N/A')}")

def main():
    """Main function demonstrating optimized agentic system"""
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-all":
            test_all_retrieval_methods_optimized()
        elif sys.argv[1] == "--test-performance":
            test_optimization_performance()
        elif sys.argv[1] == "--method" and len(sys.argv) > 2:
            # Test specific method
            company = "TATAMOTORS"
            method = sys.argv[2]
            
            print_section("OPTIMIZED SINGLE METHOD TEST", 
                         f"Company: {company}, Method: {method}", Colors.HEADER)
            
            config = OptimizedConfig(company)
            config.retrieval_method = method
            
            orchestrator = OptimizedAgenticOrchestrator(config)
            
            # Get topic choice from user - NOW USES JSON
            topic_number = get_user_topic_choice()  # This function now loads from JSON
            topic = create_sample_topic(topic_number)  # This function now loads from JSON
            
            result = orchestrator.evaluate_topic(topic)
            
            if result["success"]:
                print_section("OPTIMIZED TEST COMPLETED", 
                             f"Score: {result['scoring']['score']}/2\n" +
                             f"Total Time: {result['performance_metrics']['total_time']:.3f}s\n" +
                             f"Research Time: {result['performance_metrics']['research_time']:.3f}s\n" +
                             f"Sources: {result['research_summary']['total_sources']}\n" +
                             f"Method: {method} (Optimized)",
                             Colors.OKGREEN)
                save_results(result, config)
            else:
                print_section("TEST FAILED", result.get('error'), Colors.FAIL)
        elif sys.argv[1] == "--help":
            print_section("USAGE INSTRUCTIONS")
            print("  python main.py                                   # Interactive topic selection with hybrid method")
            print("  python main.py --test-all                        # Test all optimized methods")
            print("  python main.py --test-performance                # Performance comparison")
            print("  python main.py --method hybrid                   # Test specific optimized method with topic selection")
            print("  python main.py --method bm25                     # etc...")
            print("  python main.py --help                            # Show this help message")
        else:
            print_section("USAGE INSTRUCTIONS")
            print("  python main.py                                   # Interactive topic selection with hybrid method")
            print("  python main.py --test-all                        # Test all optimized methods")
            print("  python main.py --test-performance                # Performance comparison")
            print("  python main.py --method hybrid                   # Test specific optimized method with topic selection")
            print("  python main.py --method bm25                     # etc...")
            print("  python main.py --help                            # Show this help message")
    else:
        # Default: interactive topic selection with optimized hybrid method
        print_section("INTERACTIVE TOPIC EVALUATION", color=Colors.HEADER)
        
        config = OptimizedConfig("TATAMOTORS")
        orchestrator = OptimizedAgenticOrchestrator(config)
        
        # UPDATED PART: Now uses JSON-based topic loading with better error handling
        try:
            topic_user_choice = input("Choose a topic to evaluate (1-5) or type 'list' to see available topics: ").strip()
            if topic_user_choice.lower() == 'list':
                list_available_topics()  # This function now loads from JSON
                topic_user_choice = input("Enter topic number: ").strip()
            
            topic_number = int(topic_user_choice)
            
            # Load available topics to validate input
            available_topics = load_sample_topics()  # NEW: Load from JSON
            if topic_number not in available_topics:
                available_numbers = list(available_topics.keys())
                print(f"⚠️ Invalid topic number. Available topics: {available_numbers}. Defaulting to topic 1.")
                topic_number = 1
                
            topic = create_sample_topic(topic_number)  # This function now loads from JSON
            
        except ValueError:
            print("⚠️ Invalid input. Defaulting to topic 1 (Board Independence).")
            topic = create_sample_topic(1)  # This function now loads from JSON
        except Exception as e:
            print(f"⚠️ Error loading topic: {e}. Defaulting to topic 1.")
            topic = create_sample_topic(1)
        
        print(f"📋 Evaluating Topic: {topic.topic_name}")
        result = orchestrator.evaluate_topic(topic)
        
        if result["success"]:
            print_section("OPTIMIZED EVALUATION COMPLETED", 
                         f"Final Score: {result['scoring']['score']}/2\n" +
                         f"Total Time: {result['performance_metrics']['total_time']:.3f}s\n" +
                         f"Research Time: {result['performance_metrics']['research_time']:.3f}s\n" +
                         f"Sources: {result['research_summary']['total_sources']}\n" +
                         f"Method: {result['research_summary']['retrieval_method']} (Optimized)",
                         Colors.OKGREEN)
            
            save_results(result, config)
        else:
            print_section("EVALUATION FAILED", result.get('error'), Colors.FAIL)

if __name__ == "__main__":
    main()

