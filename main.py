import os
import json
import time
import fitz
import tempfile
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# LangChain imports
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

## Making dataclasses for structured data i,e making some fixed structures for topic definitions, questions, and answers.
@dataclass
class TopicDefinition:
    topic_name: str
    goal: str
    guidance: str
    scoring_rubric: Dict[str, str]

## Question agent should generate question in following format (also take the input as topic in above format at least initially):
@dataclass
class Question:
    text: str
    purpose: str
    priority: str

## Gemini should be enfored to return answers in the following format, I can make output guardrail without LLM as discussed with Supriya
@dataclass
class Answer:
    question: str
    answer: str
    sources: List[str]
    confidence: str
    has_citations: bool

## Main configuration class to hold all settings and API configurations
class Config:
    def __init__(self, company: str):
        self.company = company
        self.base_path = f"./data/{company}/"
        self.data_path = os.path.join(self.base_path, "98_data/")
        
        # Model settings
        self.gemini_model = "gemini-1.5-flash"
        self.ollama_model = "llama3"
        self.temperature = 0.2
        
        # Retrieval configuration
        # Here the idea is to chuck on page basis for BM25 and text basis for vector store
        # So for hybrid or vector retrieval, we will use (also/only) text chunks
        self.retrieval_method = "hybrid" # Options: "bm25", "vector", "hybrid", "direct"
        self.bm25_weight = 0.4   # Not giving user the option but to developer
        self.vector_weight = 0.6
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # PDF settings
        self.use_pdf_slices = True  #we dont send the chunk to Gemini, but rather the PDF slices from meta data, false will send chunks
        self.page_buffer = 1 # Number of pages to buffer around the relevant page
        self.max_pdf_size_mb = 20 # this limit is coming from Gemini, so we need to keep it low, but we can increase it if pro version is available
        
        # Embedding model
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  #initially tried with llama, but crazy much time it took, so falling back on this..and happy with this tbh
        
        self._setup_api()
    
    def _setup_api(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                self.genai_client = genai
            except ImportError:
                self.genai_client = None
        else:
            self.genai_client = None

## This class is heart of the data system. It uses the RAG (Retrieval-Augmented Generation) approach to process documents, create embeddings, and set up retrievers.
## so like earlier we had direct to gemini, now we can only send relevant parts of the document to Gemini, which is more efficient and cost-effective.
class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config #passing configuration class to the processor
        self.embeddings_model = None
        self.page_chunks = {}
        self.text_chunks = {}
        self.bm25_retrievers = {}
        self.vector_stores = {}
        self.hybrid_retrievers = {}
        
        self._initialize_embeddings()
        self._process_all_documents()  #we do preprocessing i.e making chunks, making pages, insert meta data etc. at the time of initialization, that way while processing the questions it is faster
        self._create_all_retrievers()  # depending on retriever selection, we create the retrievers at the time of initialization, so that we can use them later
    
    def _initialize_embeddings(self):
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")
            self.embeddings_model = None
    
    def _process_all_documents(self):
        if not os.path.exists(self.config.data_path):
            return
        
        pdf_files = [f for f in os.listdir(self.config.data_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.config.data_path, pdf_file)
            
            try:
                # Test if PDF is valid first
                test_doc = fitz.open(pdf_path)
                if len(test_doc) == 0:
                    print(f"Skipping empty PDF: {pdf_file}")
                    test_doc.close()
                    continue
                test_doc.close()
                
                # Create page chunks
                page_chunks = self._create_page_chunks(pdf_path)
                if page_chunks:
                    self.page_chunks[pdf_file] = page_chunks
                    print(f"Processed page chunks for {pdf_file}: {len(page_chunks)} pages")
                
                # Create text chunks with embeddings
                text_chunks = self._create_text_chunks_with_embeddings(pdf_path)
                if text_chunks:
                    self.text_chunks[pdf_file] = text_chunks
                    print(f"Processed text chunks for {pdf_file}: {len(text_chunks)} chunks")
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                # Remove from dictionaries if partially added
                self.page_chunks.pop(pdf_file, None)
                self.text_chunks.pop(pdf_file, None)
                continue
    
    def _create_page_chunks(self, document_path: str) -> List[Document]:
        chunks = []
        try:
            doc = fitz.open(document_path)
            file_name = os.path.basename(document_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    chunk = Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_num + 1,
                            "file_path": document_path,
                            "total_pages": len(doc),
                            "chunk_type": "page"
                        }
                    )
                    chunks.append(chunk)
            
            doc.close()
            
        except Exception as e:
            print(f"Error creating page chunks: {e}")
        
        return chunks
    
    def _create_text_chunks_with_embeddings(self, document_path: str) -> List[Document]:
        chunks = []
        try:
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata['file_path'] = document_path
                doc.metadata['source'] = os.path.basename(document_path)
                chunks.append(doc)
            
        except Exception as e:
            print(f"Error creating text chunks: {e}")
        
        return chunks
    
    def _create_all_retrievers(self):
        # Only process files that were successfully loaded
        valid_files = set(self.page_chunks.keys()) & set(self.text_chunks.keys())
        
        for pdf_file in valid_files:
            try:
                # Create BM25 retriever from page chunks
                if pdf_file in self.page_chunks and self.page_chunks[pdf_file]:
                    bm25_retriever = BM25Retriever.from_documents(self.page_chunks[pdf_file])
                    self.bm25_retrievers[pdf_file] = bm25_retriever
                    print(f"Created BM25 retriever for {pdf_file}")
                
                # Create vector store from text chunks
                if pdf_file in self.text_chunks and self.text_chunks[pdf_file] and self.embeddings_model:
                    try:
                        vector_store = Chroma.from_documents(
                            documents=self.text_chunks[pdf_file],
                            embedding=self.embeddings_model
                        )
                        self.vector_stores[pdf_file] = vector_store
                        print(f"Created vector store for {pdf_file}")
                    except Exception as e:
                        print(f"Error creating vector store for {pdf_file}: {e}")
                
                # Create hybrid retriever
                if pdf_file in self.bm25_retrievers and pdf_file in self.vector_stores:
                    try:
                        vector_retriever = self.vector_stores[pdf_file].as_retriever()
                        
                        hybrid_retriever = EnsembleRetriever(
                            retrievers=[self.bm25_retrievers[pdf_file], vector_retriever],
                            weights=[self.config.bm25_weight, self.config.vector_weight]
                        )
                        
                        self.hybrid_retrievers[pdf_file] = hybrid_retriever
                        print(f"Created hybrid retriever for {pdf_file}")
                    except Exception as e:
                        print(f"Error creating hybrid retriever for {pdf_file}: {e}")
                        
            except Exception as e:
                print(f"Error creating retrievers for {pdf_file}: {e}")
                continue
        
        print(f"Successfully created retrievers for {len(valid_files)} files")
    
    def get_retriever(self, pdf_file: str, method: str = None):
        if method is None:
            method = self.config.retrieval_method
        
        if method == "bm25":
            return self.bm25_retrievers.get(pdf_file)
        elif method == "vector":
            if pdf_file in self.vector_stores:
                return self.vector_stores[pdf_file].as_retriever()
        elif method == "hybrid":
            return self.hybrid_retrievers.get(pdf_file)
        elif method == "direct":
            return None
        
        return None

## This generates initial question based on the topic definition.
## and follow up questions can be generated based on the answers received from the research agent.
## it is iteratively used to refine the questions based on the answers received.
## This generates initial question based on the topic definition.
## and follow up questions can be generated based on the answers received from the research agent.
## it is iteratively used to refine the questions based on the answers received.
class QuestionAgent:
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._get_llm()
    
    def _get_llm(self):
        if self.config.genai_client:
            return self.config.genai_client.GenerativeModel(self.config.gemini_model)
        else:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=self.config.ollama_model, temperature=self.config.temperature)
    
    def generate_initial_question(self, topic: TopicDefinition) -> Question:
        
        ## when you dont have LLM, then you can not generate question, so this is a fallback
        ## some may ask, if there is no LLM then why to even bother about the question, anyways answer is going to come from llm
        ## well, question agent can have different LLM, them reaserch agent, which may have llm available.
        if not self.llm: 
            return Question(
                text=f"What information is available about {topic.topic_name}?",
                purpose="Fallback question",
                priority="high"
            )
        
        prompt = f"""
        Analyze this topic and create ONE key question that will help distinguish between the scoring levels.
        
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
                return Question(
                    text=result.get("question", ""),
                    purpose=result.get("purpose", ""),
                    priority=result.get("priority", "high")
                )
        except Exception as e:
            print(f"Question generation error: {e}")
        
        return Question(
            text=f"What specific information about {topic.topic_name} is disclosed in the documents?",
            purpose="Fallback question",
            priority="high"
        )
    
    def generate_follow_up_question(self, topic: TopicDefinition, existing_answers: List[Answer]) -> Optional[Question]:
        """Generate follow-up question based on gaps in existing answers"""
        
        if not self.llm:
            print("No LLM available for follow-up questions")
            return None
        
        # Prepare summary of existing answers
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
                    print(f"Follow-up needed: {result.get('gap_identified', 'Gap not specified')}")
                    return Question(
                        text=result["question"],
                        purpose=result.get("purpose", ""),
                        priority=result.get("priority", "medium")
                    )
                else:
                    print("No follow-up needed - sufficient information for scoring")
                    return None
        
        except Exception as e:
            print(f"Follow-up question generation error: {e}")
            return None
    
    def check_information_sufficiency(self, topic: TopicDefinition, existing_answers: List[Answer]) -> bool:
        """Check if we have sufficient information to proceed with scoring"""
        
        if not self.llm:
            # If no LLM, assume we need more info unless we have multiple answers
            return len(existing_answers) >= 2
        
        # Prepare summary of existing answers
        answer_context = "\n".join([
            f"Q: {ans.question}\nA: {ans.answer}\nSources: {', '.join(ans.sources)}\nConfidence: {ans.confidence}"
            for ans in existing_answers
        ])
        
        prompt = f"""
        You are evaluating if there is sufficient information to score a corporate governance topic.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        EXISTING RESEARCH:
        {answer_context}
        
        Question: Based on the existing research, do you have sufficient information to confidently apply the scoring rubric and assign a score (0, 1, or 2)?
        
        Consider:
        1. Can you distinguish between the different scoring levels with the current evidence?
        2. Are there key criteria in the rubric that are not addressed?
        3. Is the evidence quality sufficient for confident scoring?
        
        Respond in JSON format:
        {{
            "sufficient": true/false,
            "reason": "explanation of why information is sufficient or insufficient",
            "confidence_level": "high/medium/low"
        }}
        
        Be conservative - only return true if you can confidently score.
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
                
                is_sufficient = result.get("sufficient", False)
                reason = result.get("reason", "No reason provided")
                
                if is_sufficient:
                    print(f"Information sufficient for scoring: {reason}")
                else:
                    print(f"Information insufficient: {reason}")
                
                return is_sufficient
        
        except Exception as e:
            print(f"Sufficiency check error: {e}")
            return False
        
        # Default to insufficient if we can't determine
        return False

## This agent handles the research process, querying documents and generating answers based on the retrieved information.
## it leverage the document processor to find relevant chunks of text or PDF slices.
## this guy coordinnates with gemini to get answers.
class ResearchAgent:
    def __init__(self, config: Config):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.llm = self._get_llm()
    
    def _get_llm(self):
        if self.config.genai_client:
            return self.config.genai_client.GenerativeModel(self.config.gemini_model)
        else:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=self.config.ollama_model, temperature=self.config.temperature)
    
    def research_question(self, question: str) -> Answer:
        if not self.llm:
            return Answer(
                question=question,
                answer="No LLM available",
                sources=[],
                confidence="low",
                has_citations=False
            )
        
        # Find relevant chunks
        relevant_chunks = self._find_relevant_chunks(question)
        
        if not relevant_chunks:
            return self._fallback_to_direct_method(question)
        
        # Process with chunks or PDF slices
        if self.config.use_pdf_slices and self.config.genai_client:
            answer_text = self._query_with_pdf_slices(question, relevant_chunks)
        else:
            answer_text = self._query_with_chunks(question, relevant_chunks)
        
        sources = self._extract_sources_from_chunks(relevant_chunks)
        has_citations = self._has_source_citations(answer_text)
        confidence = self._assess_confidence(answer_text, relevant_chunks)
        
        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            has_citations=has_citations
        )
    
    def _find_relevant_chunks(self, question: str) -> List[Document]:
        all_relevant_chunks = []
        
        for pdf_file in self.document_processor.page_chunks.keys():
            if self.config.retrieval_method == "direct":
                continue
            
            retriever = self.document_processor.get_retriever(pdf_file, self.config.retrieval_method)
            if not retriever:
                continue
            
            try:
                relevant_chunks = retriever.invoke(question)
                
                for chunk in relevant_chunks:
                    if 'file_path' not in chunk.metadata:
                        chunk.metadata['file_path'] = os.path.join(self.config.data_path, pdf_file)
                
                all_relevant_chunks.extend(relevant_chunks)
                
            except Exception as e:
                print(f"Error retrieving from {pdf_file}: {e}")
                continue
        
        return all_relevant_chunks
    
    def _query_with_pdf_slices(self, question: str, chunks: List[Document]) -> str:
        try:
            # Extract file paths and page numbers
            pdf_slices = []
            
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                    metadata = chunk.metadata
                    
                    file_path = None
                    for key in ['file_path', 'source', 'path']:
                        if key in metadata and metadata[key]:
                            file_path = metadata[key]
                            break
                    
                    page_num = None
                    for key in ['page', 'page_number', 'page_num']:
                        if key in metadata and metadata[key]:
                            try:
                                page_num = int(metadata[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    if file_path and page_num:
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(self.config.data_path, file_path)
                        
                        if os.path.exists(file_path):
                            pdf_slices.append({"file_path": file_path, "page": page_num})
            
            if not pdf_slices:
                return self._query_with_chunks(question, chunks)
            
            # Group slices by file and add page buffers
            files_to_pages = {}
            for s in pdf_slices:
                file_path = s['file_path']
                page = s['page']
                
                if file_path not in files_to_pages:
                    files_to_pages[file_path] = set()
                
                for offset in range(-self.config.page_buffer, self.config.page_buffer + 1):
                    buffered_page = page + offset
                    if buffered_page > 0:
                        files_to_pages[file_path].add(buffered_page)
            
            # Create temporary PDF with relevant pages
            output_pdf = fitz.open()
            
            for file_path, pages in files_to_pages.items():
                try:
                    doc = fitz.open(file_path)
                    total_pages = len(doc)
                    
                    valid_pages = []
                    for p in sorted(pages):
                        if 1 <= p <= total_pages:
                            valid_pages.append(p - 1)  # Convert to 0-indexed
                    
                    for page_num in valid_pages:
                        output_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    
                    doc.close()
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
            if output_pdf.page_count == 0:
                output_pdf.close()
                return self._query_with_chunks(question, chunks)
            
            # Save temporary PDF and query Gemini
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"temp_slice_{int(time.time())}.pdf")
            output_pdf.save(temp_pdf_path)
            output_pdf.close()
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            with open(temp_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    question
                ]
            )
            
            result = response.text
            
            # Clean up
            try:
                os.remove(temp_pdf_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            print(f"Error in PDF slice query: {e}")
            return self._query_with_chunks(question, chunks)
    
    def _query_with_chunks(self, question: str, chunks: List[Document]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk.metadata.get('source', 'Unknown')}, Page: {chunk.metadata.get('page', 'Unknown')}"
            chunk_text = chunk.page_content[:1500]
            context_parts.append(f"[Chunk {i+1}] {source_info}\n{chunk_text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""
        QUESTION: {question}
        
        DOCUMENT EXCERPTS:
        {context}
        
        Instructions:
        1. Answer based ONLY on the provided excerpts
        2. Include specific source citations (page numbers and document names)
        3. Be precise and factual
        
        Provide a comprehensive answer with proper source citations.
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                result = response.text
            else:
                result = self.llm.invoke(prompt)
            
            return result
                
        except Exception as e:
            return f"Error querying documents: {str(e)}"
    
    def _fallback_to_direct_method(self, question: str) -> Answer:
        pdf_files = list(self.document_processor.page_chunks.keys())
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.config.data_path, pdf_file)
            
            try:
                answer = self._query_entire_document(pdf_path, question)
                if answer:
                    return Answer(
                        question=question,
                        answer=answer,
                        sources=[pdf_file],
                        confidence="medium",
                        has_citations=self._has_source_citations(answer)
                    )
            except Exception as e:
                continue
        
        return Answer(
            question=question,
            answer="Could not find relevant information",
            sources=[],
            confidence="low",
            has_citations=False
        )
    
    def _query_entire_document(self, document_path: str, question: str) -> str:
        try:
            if not self.config.genai_client:
                return None
            
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                return None
            
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
            
            return response.text
            
        except Exception as e:
            return None
    
    def _extract_sources_from_chunks(self, chunks: List[Document]) -> List[str]:
        sources = []
        for chunk in chunks:
            metadata = chunk.metadata
            source_info = f"Page {metadata.get('page', 'Unknown')} ({metadata.get('source', 'Unknown')})"
            if source_info not in sources:
                sources.append(source_info)
        return sources
    
    def _has_source_citations(self, answer_text: str) -> bool:
        citation_patterns = ['page', 'source:', 'according to', 'document', 'pp.', 'from']
        answer_lower = answer_text.lower()
        return any(pattern in answer_lower for pattern in citation_patterns)
    
    def _assess_confidence(self, answer_text: str, chunks: List[Document]) -> str:
        if len(chunks) >= 5 and len(answer_text) > 200 and self._has_source_citations(answer_text):
            return "high"
        elif len(chunks) >= 3 and len(answer_text) > 100:
            return "medium"
        else:
            return "low"

## Scoring agent takes all work done by questionagent and research agent, and scores the topic based on the answers received.
class ScoringAgent:
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._get_llm()
    
    def _get_llm(self):
        if self.config.genai_client:
            return self.config.genai_client.GenerativeModel(self.config.gemini_model)
        else:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=self.config.ollama_model, temperature=0.1)
    
    def score_topic(self, topic: TopicDefinition, answers: List[Answer]) -> Dict[str, Any]:
        if not self.llm:
            return {
                "score": 0,
                "justification": "No LLM available for scoring",
                "confidence": "low",
                "evidence_quality": "poor"
            }
        
        evidence_summary = self._prepare_evidence_summary(answers)
        
        prompt = f"""
        You are scoring a corporate governance topic based on collected research evidence.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        RESEARCH EVIDENCE:
        {evidence_summary}
        
        Instructions:
        1. Evaluate the evidence against each scoring level in the rubric
        2. Assign a score (0, 1, or 2) based on which level best matches the evidence
        3. Provide detailed justification with specific references to the evidence
        4. Preserve all source citations from the evidence in your justification
        
        Respond in JSON format:
        {{
            "score": 0/1/2,
            "justification": "Detailed justification with source citations",
            "evidence_quality": "excellent/good/fair/poor",
            "key_findings": ["list of key findings that influenced the score"]
        }}
        
        Be objective and base your score strictly on the evidence provided.
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
                
                score = result.get("score", 0)
                if score not in [0, 1, 2]:
                    score = 0
                
                confidence = self._assess_scoring_confidence(answers)
                
                return {
                    "score": score,
                    "justification": result.get("justification", "No justification provided"),
                    "evidence_quality": result.get("evidence_quality", "fair"),
                    "key_findings": result.get("key_findings", []),
                    "confidence": confidence
                }
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Scoring error: {e}")
            return {
                "score": 0,
                "justification": f"Scoring failed: {str(e)}",
                "confidence": "low",
                "evidence_quality": "poor"
            }
    
    def _prepare_evidence_summary(self, answers: List[Answer]) -> str:
        summary_parts = []
        
        for i, answer in enumerate(answers, 1):
            part = f"""
            EVIDENCE {i}:
            Question: {answer.question}
            Answer: {answer.answer}
            Sources: {', '.join(answer.sources)}
            Confidence: {answer.confidence}
            ---
            """
            summary_parts.append(part)
        
        return "\n".join(summary_parts)
    
    def _assess_scoring_confidence(self, answers: List[Answer]) -> str:
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

## it determines the flow of the entire evaluation process, coordinating between question generation, research, and scoring.
class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.question_agent = QuestionAgent(config)
        self.research_agent = ResearchAgent(config)
        self.scoring_agent = ScoringAgent(config)
        self.answers = []
        self.max_iterations = 3
    
    def evaluate_topic(self, topic: TopicDefinition) -> Dict[str, Any]:
        print(f"Evaluating topic: {topic.topic_name}")
        
        total_start_time = time.time()
        self.answers = []
        
        # Generate initial question
        current_question = self.question_agent.generate_initial_question(topic)
        print(f"Initial question: {current_question.text}")
        print(f"Purpose: {current_question.purpose}")
        
        # Iterative research process
        iteration = 0
        total_research_time = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Research iteration {iteration} ---")
            print(f"Current question: {current_question.text}")
            
            # Research the current question
            research_start = time.time()
            answer = self.research_agent.research_question(current_question.text)
            research_time = time.time() - research_start
            total_research_time += research_time
            
            self.answers.append(answer)
            print(f"Answer confidence: {answer.confidence}")
            print(f"Has citations: {answer.has_citations}")
            print(f"Sources count: {len(answer.sources)}")
            
            # Check if we have sufficient information to proceed with scoring
            has_sufficient_info = self.question_agent.check_information_sufficiency(topic, self.answers)
            
            if has_sufficient_info:
                print("✅ Sufficient information collected - proceeding to scoring")
                break
            
            # Generate follow-up question if we need more information
            if iteration < self.max_iterations:
                print("🔍 Checking if follow-up question needed...")
                follow_up_question = self.question_agent.generate_follow_up_question(topic, self.answers)
                
                if follow_up_question is None:
                    print("✅ No follow-up question needed - proceeding to scoring")
                    break
                else:
                    print(f"➡️ Follow-up question generated: {follow_up_question.text}")
                    current_question = follow_up_question
            else:
                print("⏰ Maximum iterations reached - proceeding to scoring")
        
        # Final scoring
        print(f"\n--- Final scoring after {iteration} iterations ---")
        scoring_start = time.time()
        scoring_result = self.scoring_agent.score_topic(topic, self.answers)
        scoring_time = time.time() - scoring_start
        
        total_time = time.time() - total_start_time
        
        # Calculate performance metrics
        avg_research_per_iteration = total_research_time / iteration if iteration > 0 else 0
        
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
                "retrieval_method": self.config.retrieval_method,
                "total_sources": len(set([s for ans in self.answers for s in ans.sources])),
                "early_termination": iteration < self.max_iterations,
                "termination_reason": "sufficient_information" if iteration < self.max_iterations else "max_iterations"
            },
            "performance_metrics": {
                "total_time": total_time,
                "research_time": total_research_time,
                "scoring_time": scoring_time,
                "avg_research_per_iteration": avg_research_per_iteration
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
        
        print(f"\n🎯 Evaluation completed - Score: {scoring_result.get('score', 'N/A')}/2")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"📊 Research iterations: {iteration}")
        print(f"📈 Evidence quality: {scoring_result.get('evidence_quality', 'unknown')}")
        
        return result
    
## Usually to be taken from user, but for testing we can create a sample topic
def create_sample_topic() -> TopicDefinition:
    return TopicDefinition(
        topic_name="Board Independence",
        goal="To assess if the board have directors with permanent board seats",
        guidance="You need to look for the corporate governance report. Find the reappointment date for each board members. If the reappointment date is either not provided or older than 5 years (i.e some date before 2019), then you need to check appointment date. If appointment date is also older than 5 years (i.e before 2019), mark that board member as permanent. Give list of board members and whether or not they are permanent. In other words, either of appointment date or reappointment date should be within last 5 years.",
        scoring_rubric={
            "0": "if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders.",
            "1": "if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so.",
            "2": "if All directors are marked as non-permanent board members"
        }
    )

## taking the company names, configuring and handing it over to orchestrator
def main():
    # Test the system
    config = Config("PAYTM")
    orchestrator = Orchestrator(config)
    topic = create_sample_topic()
    
    result = orchestrator.evaluate_topic(topic)
    
    if result["success"]:
        print("\nResults:")
        print(f"Final Score: {result['scoring']['score']}/2")
        print(f"Total Time: {result['performance_metrics']['total_time']:.2f}s")
        print(f"Evidence Quality: {result['scoring']['evidence_quality']}")
        print(f"Sources: {result['research_summary']['total_sources']}")
        
        # Save results
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Results saved to results/evaluation_results.json")
    else:
        print(f"Evaluation failed: {result.get('error')}")

if __name__ == "__main__":
    main()