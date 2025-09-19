import os
import io
import re
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import streamlit as st

# Parsing
import pandas as pd
from PIL import Image

# Optional dependencies handled gracefully
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pptx import Presentation
except Exception:
    Presentation = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# LangChain imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_chroma import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except Exception as e:
    st.error(f"LangChain dependencies not installed: {e}")
    st.stop()

# Chroma vector database
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    st.error("ChromaDB not installed. Run: pip install chromadb")
    st.stop()

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# Utilities and Configuration
# -----------------------------

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap."""
    if not text.strip():
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        if end >= len(words):
            break
            
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def generate_answer(query: str, contexts: List) -> str:
    """Generate answer using OpenAI or fallback to simple context summary."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return "Please provide an OpenAI API key to enable answer generation."
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Prepare context from search results
        context_text = ""
        for i, ctx in enumerate(contexts):
            context_text += f"[Context {i+1}]: {ctx.text}\n\n"
        
        prompt = f"""Based on the following context, answer the user's question accurately and concisely.

Context:
{context_text}

Question: {query}

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    question: str
    answer: str
    sources: List[str]
    timestamp: datetime


@dataclass
class DocumentChunk:
    """Represents a document chunk with text and metadata."""
    text: str
    meta: Dict


class DocumentProcessor:
    """Enhanced document processing with LangChain integration."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, texts_and_metadata: List[Tuple[str, Dict]]) -> List[Document]:
        """Process texts into LangChain documents with metadata."""
        documents = []
        for text, metadata in texts_and_metadata:
            if not text.strip():
                continue
            
            # Create document with metadata
            doc = Document(
                page_content=clean_text(text),
                metadata={
                    **metadata,
                    'processed_at': datetime.now().isoformat(),
                    'doc_id': str(uuid.uuid4())
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{chunk.metadata['doc_id']}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
            
            documents.extend(chunks)
        
        return documents


# -----------------------------
# Loaders for different file types
# -----------------------------

def load_pdf(file: io.BytesIO) -> str:
    if pdfplumber is None:
        st.warning("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    all_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            all_text.append(page.extract_text() or "")
    return "\n".join(all_text)


def load_docx(file: io.BytesIO) -> str:
    if docx is None:
        st.warning("python-docx not installed. Run: pip install python-docx")
        return ""
    d = docx.Document(file)
    return "\n".join([p.text for p in d.paragraphs])


def load_pptx(file: io.BytesIO) -> str:
    if Presentation is None:
        st.warning("python-pptx not installed. Run: pip install python-pptx")
        return ""
    prs = Presentation(file)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)


def load_image(file: io.BytesIO) -> str:
    if pytesseract is None:
        st.warning("pytesseract not installed (and requires Tesseract binary). Skipping OCR.")
        return ""
    try:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def load_txt(file: io.BytesIO) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return file.read().decode("latin-1", errors="ignore")


def load_md(file: io.BytesIO) -> str:
    return load_txt(file)


def load_csv(file: io.BytesIO) -> str:
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_excel(file)
    # Convert to a readable text table
    return df.to_csv(index=False)


LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".txt": load_txt,
    ".md": load_md,
    ".csv": load_csv,
    ".xlsx": load_csv,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
}


# -----------------------------
# Enhanced Vector Store with Chroma
# -----------------------------

class EnhancedVectorStore:
    """Enhanced vector store using Chroma with LangChain integration."""
    
    def __init__(self, collection_name: str = "documents", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", use_openai_embeddings: bool = False):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_openai_embeddings = use_openai_embeddings
        self.vectorstore = None
        self.embeddings = None
        self._initialize_embeddings()
        self._initialize_vectorstore()
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        if self.use_openai_embeddings and os.getenv("OPENAI_API_KEY"):
            try:
                self.embeddings = OpenAIEmbeddings()
            except Exception as e:
                st.warning(f"Failed to initialize OpenAI embeddings: {e}. Falling back to SentenceTransformer.")
                self.embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        else:
            self.embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)
    
    def _initialize_vectorstore(self):
        """Initialize Chroma vector store."""
        try:
            # Create persistent Chroma client
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            st.error(f"Failed to initialize Chroma vector store: {e}")
            st.stop()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        try:
            self.vectorstore.add_documents(documents)
        except Exception as e:
            st.error(f"Failed to add documents to vector store: {e}")
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """Search for similar documents with scores."""
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            st.error(f"Failed to search vector store: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Dict = None):
        """Get a retriever for use with LangChain chains."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.vectorstore._collection.count()
        except:
            return 0


# -----------------------------
# VectorIndex Class (Wrapper for EnhancedVectorStore)
# -----------------------------

class VectorIndex:
    """Vector index wrapper for compatibility with existing Streamlit code."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.vectorstore = EnhancedVectorStore(
            collection_name="documents",
            embedding_model=model_name,
            use_openai_embeddings=False
        )
        self.document_processor = DocumentProcessor()
        # Add embeddings attribute for compatibility
        self.embeddings = self.vectorstore.embeddings
    
    def add_documents(self, texts_and_metadata: List[Tuple[str, Dict]]):
        """Add documents to the vector index."""
        documents = self.document_processor.process_documents(texts_and_metadata)
        self.vectorstore.add_documents(documents)
    
    def add(self, chunks: List[DocumentChunk]):
        """Add DocumentChunk objects to the vector index."""
        texts_and_metadata = [(chunk.text, chunk.meta) for chunk in chunks]
        self.add_documents(texts_and_metadata)
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """Search for similar documents with scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: Dict = None):
        """Get a retriever for use with LangChain chains."""
        return self.vectorstore.get_retriever(search_kwargs)
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.vectorstore.get_collection_count()


# -----------------------------
# Enhanced Q&A System with LangChain
# -----------------------------

class ConversationalQASystem:
    """Enhanced Q&A system with conversation memory and better source citations."""
    
    def __init__(self, vector_store: EnhancedVectorStore, memory_window: int = 10):
        self.vector_store = vector_store
        self.memory_window = memory_window
        self.conversation_memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        self._initialize_qa_chain()
    
    def _initialize_qa_chain(self):
        """Initialize the conversational QA chain."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            try:
                # Use OpenAI for generation
                llm = ChatOpenAI(
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.2,
                    api_key=api_key
                )
                
                # Custom prompt template with better source citations
                custom_prompt = PromptTemplate(
                    template="""You are an intelligent document assistant. Use the provided context to answer the user's question accurately and concisely.

IMPORTANT INSTRUCTIONS:
1. Always cite your sources using the format [Source: filename] when referencing information
2. If information comes from multiple sources, cite all relevant sources
3. If you cannot find the answer in the provided context, clearly state that you don't have enough information
4. Provide specific, actionable answers when possible
5. Consider the conversation history for context

Chat History:
{chat_history}

Context from documents:
{context}

Human Question: {question}
""".strip(),
                    input_variables=["chat_history", "context", "question"]
                )
                
                # Create the conversational retrieval chain
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.vector_index.vectorstore.as_retriever(search_kwargs={"k": 5}),
                    combine_docs_chain_kwargs={"prompt": custom_prompt},
                    return_source_documents=True,
                    verbose=True
                )
                
            except Exception as e:
                st.error(f"Failed to initialize OpenAI: {e}")
                self.qa_chain = None
        else:
            self.qa_chain = None

    def get_answer(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Dict:
        """Get answer using the QA chain."""
        if not self.qa_chain:
            return {
                "answer": "Please provide an OpenAI API key to enable answer generation.",
                "source_documents": []
            }
        
        if chat_history is None:
            chat_history = []
        
        try:
            result = self.qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            return result
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "source_documents": []
            }

# Streamlit UI
st.title("🧠 Intelligent Document Assistant — Prototype")

with st.sidebar:
    st.header("Settings")
    st.markdown("Provide your OpenAI API key to enable answer generation (optional).")
    api_key_input = st.text_input("OPENAI_API_KEY", type="password", help="Used locally in your session only.")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
        ],
        index=0,
        help="MiniLM-L6-v2 is fast and good enough for prototypes.",
    )

    chunk_size = st.slider("Chunk size (words)", 200, 1200, 500, step=50)
    overlap = st.slider("Chunk overlap (words)", 0, 400, 100, step=10)

    top_k = st.slider("Top-K chunks to retrieve", 1, 10, 5)

st.markdown(
    "Upload multiple files (PDF, DOCX, PPTX, TXT, MD, CSV/XLSX, PNG/JPG). I'll parse, chunk, and index them for Q&A."
)

uploaded_files = st.file_uploader(
    "Upload documents", type=list(LOADER_MAP.keys()), accept_multiple_files=True
)

index_state = st.session_state.get("vindex")
if index_state is None:
    st.session_state["vindex"] = VectorIndex(model_name=model_name)
    index_state = st.session_state["vindex"]

# If user changed embedding model, reset the index
if index_state.model_name != model_name:
    st.info("Embedding model changed — resetting the index.")
    st.session_state["vindex"] = VectorIndex(model_name=model_name)
    index_state = st.session_state["vindex"]

col_left, col_right = st.columns([2, 1])

with col_left:
    if uploaded_files:
        new_chunks: List[DocumentChunk] = []
        for up in uploaded_files:
            name = up.name
            suffix = os.path.splitext(name)[1].lower()
            loader = LOADER_MAP.get(suffix)
            if not loader:
                st.warning(f"Unsupported file type: {suffix}")
                continue
            try:
                text = loader(up)
            except Exception as e:
                st.error(f"Failed to read {name}: {e}")
                text = ""
            text = clean_text(text)
            if not text:
                st.warning(f"No text extracted from {name}")
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for ch in chunks:
                new_chunks.append(DocumentChunk(text=ch, meta={"source": name}))
        if new_chunks:
            with st.spinner("Indexing new chunks ..."):
                index_state.add(new_chunks)
            st.success(f"Indexed {len(new_chunks)} chunks from {len(uploaded_files)} file(s).")

    st.subheader("Ask questions about your documents")
    query = st.text_input("Your question")
    ask = st.button("🔎 Retrieve & Answer", type="primary", use_container_width=True)

    if ask and query.strip():
        with st.spinner("Retrieving relevant chunks ..."):
            results = index_state.query(query, top_k=top_k)
        if not results:
            st.warning("Index is empty or nothing found. Upload files first.")
        else:
            contexts = [dc for sim, dc in results]
            st.write("### Top matches")
            for i, (sim, dc) in enumerate(results, start=1):
                with st.expander(f"[S{i}] {dc.meta.get('source','unknown')} — similarity {sim:.3f}"):
                    st.write(dc.text)

            with st.spinner("Generating answer ..."):
                answer = generate_answer(query, contexts)
            st.write("### Answer")
            st.write(answer)

with col_right:
    st.subheader("Index status")
    n = index_state.get_collection_count()
    st.metric("Chunks indexed", n)
    st.caption("Re-upload files anytime to add more chunks. Changing the embedding model resets the index.")

    st.divider()
    st.subheader("Quick Tips")
    st.markdown(
        """
        - Provide an **OpenAI API key** in the sidebar to enable LLM-generated answers.\
        - Tune **chunk size/overlap** for your content: smaller for slides, larger for reports.\
        - OCR for images requires **pytesseract** and the **Tesseract** binary installed locally.\
        - Supported types: PDF, DOCX, PPTX, TXT, MD, CSV/XLSX, PNG/JPG.
        """
    )

st.divider()
with st.expander("📦 Setup (requirements.txt)"):
    st.code(
        """
        streamlit
        pdfplumber
        python-docx
        python-pptx
        pandas
        pillow
        sentence-transformers
        scikit-learn
        openai>=1.0.0
        pytesseract
        """.strip(),
        language="text",
    )

st.caption("Prototype built for multi-format ingestion, semantic retrieval, and optional LLM answers. Extend with persistent storage (FAISS/Chroma), auth, and background indexing for production.")