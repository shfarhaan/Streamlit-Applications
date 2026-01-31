"""
RAG (Retrieval-Augmented Generation) Document Q&A System

This application demonstrates a modern RAG architecture for question-answering
over your documents. It uses embeddings for semantic search and LLM for generation.

Note: This demo uses a lightweight approach with sentence transformers and 
can be adapted to use OpenAI, Anthropic, or other LLM APIs.
"""

import streamlit as st
import numpy as np
from typing import List, Tuple
import re

# For this demo, we'll use a simple approach that can be enhanced with actual LLM APIs
# In production, you would use: OpenAI, Anthropic Claude, or open-source models

class SimpleRAGSystem:
    """A simple RAG system for demonstration"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        
    def add_document(self, text: str):
        """Add a document to the knowledge base"""
        # Split into chunks for better retrieval
        chunks = self._chunk_text(text)
        self.documents.extend(chunks)
        
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create a simple word-based embedding (for demo purposes)"""
        # In production, use sentence-transformers or OpenAI embeddings
        words = set(text.lower().split())
        # Simple bag-of-words representation
        vocab = set()
        for doc in self.documents:
            vocab.update(doc.lower().split())
        
        vocab_list = sorted(list(vocab))
        embedding = np.zeros(len(vocab_list))
        
        for i, word in enumerate(vocab_list):
            if word in words:
                embedding[i] = 1
        
        return embedding
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant documents"""
        if not self.documents:
            return []
        
        query_embedding = self._simple_embedding(query)
        
        # Calculate similarity scores
        scores = []
        for doc in self.documents:
            doc_embedding = self._simple_embedding(doc)
            # Cosine similarity
            if np.linalg.norm(query_embedding) > 0 and np.linalg.norm(doc_embedding) > 0:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
            else:
                similarity = 0
            scores.append((doc, similarity))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate an answer based on retrieved context"""
        # In production, this would call an LLM API
        # For demo, we'll do simple extraction
        
        if not context:
            return "I don't have enough information to answer that question."
        
        # Simple answer generation: find sentences containing query keywords
        query_keywords = set(query.lower().split())
        relevant_sentences = []
        
        for doc in context:
            sentences = re.split(r'[.!?]+', doc)
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = query_keywords & sentence_words
                if len(overlap) >= 2:  # At least 2 keyword matches
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ' '.join(relevant_sentences[:3])
        else:
            return f"Based on the documents: {context[0][:200]}..."


def main():
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Document Q&A System")
    st.markdown("""
    ### Retrieval-Augmented Generation (RAG) Demo
    
    This application demonstrates a modern RAG architecture:
    1. **Document Ingestion**: Upload and process documents
    2. **Semantic Search**: Find relevant information using embeddings
    3. **Answer Generation**: Generate answers based on retrieved context
    
    **Note**: This is a simplified demo. In production, you would use:
    - Sentence-transformers or OpenAI embeddings for better semantic search
    - OpenAI GPT-4, Anthropic Claude, or other LLMs for generation
    - Vector databases (Pinecone, Weaviate, ChromaDB) for scalability
    """)
    
    # Initialize RAG system in session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SimpleRAGSystem()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 5, 3)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### How it works:
    1. Upload your documents
    2. Documents are split into chunks
    3. Ask questions about the content
    4. System retrieves relevant chunks
    5. Generates contextual answers
    """)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "💬 Ask Questions", "📊 System Info"])
    
    with tab1:
        st.header("Upload Your Documents")
        
        # Sample documents option
        use_sample = st.checkbox("Use sample documents", value=True)
        
        if use_sample:
            st.info("Using sample documents about AI and Machine Learning")
            sample_docs = [
                """
                Artificial Intelligence (AI) is the simulation of human intelligence by machines.
                Machine learning is a subset of AI that enables computers to learn from data.
                Deep learning uses neural networks with multiple layers to process complex patterns.
                Natural Language Processing (NLP) allows computers to understand and generate human language.
                Computer vision enables machines to interpret and analyze visual information.
                """,
                """
                Retrieval-Augmented Generation (RAG) is a technique that enhances language models
                by retrieving relevant information from a knowledge base before generating responses.
                This approach combines the benefits of retrieval systems and generative models.
                RAG systems typically use vector databases to store and retrieve document embeddings.
                The process involves encoding documents, storing them, and retrieving relevant context
                for each query before generating a response.
                """,
                """
                Large Language Models (LLMs) like GPT-4, Claude, and others are trained on vast amounts
                of text data. They can perform various tasks like translation, summarization, and 
                question-answering. However, they can hallucinate or provide outdated information.
                RAG helps mitigate these issues by grounding responses in retrieved factual content.
                Fine-tuning and prompt engineering are important techniques for improving LLM performance.
                """
            ]
            
            if st.button("Load Sample Documents"):
                st.session_state.rag_system.documents = []
                for doc in sample_docs:
                    st.session_state.rag_system.add_document(doc)
                st.success(f"Loaded {len(st.session_state.rag_system.documents)} document chunks!")
        
        # File upload
        st.markdown("---")
        uploaded_file = st.file_uploader("Or upload a text file", type=['txt'])
        
        if uploaded_file is not None:
            text_content = uploaded_file.read().decode('utf-8')
            if st.button("Process Uploaded File"):
                st.session_state.rag_system.add_document(text_content)
                st.success("Document processed and added to knowledge base!")
        
        # Manual text input
        st.markdown("---")
        manual_text = st.text_area("Or paste text directly:", height=200)
        if st.button("Add Text to Knowledge Base"):
            if manual_text:
                st.session_state.rag_system.add_document(manual_text)
                st.success("Text added to knowledge base!")
        
        # Show current knowledge base stats
        if st.session_state.rag_system.documents:
            st.markdown("---")
            st.metric("Total Document Chunks", len(st.session_state.rag_system.documents))
    
    with tab2:
        st.header("Ask Questions")
        
        if not st.session_state.rag_system.documents:
            st.warning("⚠️ Please upload documents first in the 'Upload Documents' tab!")
        else:
            # Question input
            question = st.text_input("Enter your question:", 
                                    placeholder="e.g., What is RAG?")
            
            if st.button("Get Answer", type="primary"):
                if question:
                    with st.spinner("Searching knowledge base and generating answer..."):
                        # Retrieve relevant documents
                        retrieved_docs = st.session_state.rag_system.retrieve(question, top_k)
                        
                        if retrieved_docs:
                            # Show retrieved context
                            with st.expander("📚 Retrieved Context", expanded=False):
                                for i, (doc, score) in enumerate(retrieved_docs, 1):
                                    st.markdown(f"**Chunk {i}** (Relevance: {score:.2f})")
                                    st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                                    st.markdown("---")
                            
                            # Generate answer
                            context = [doc for doc, _ in retrieved_docs]
                            answer = st.session_state.rag_system.generate_answer(question, context)
                            
                            # Display answer
                            st.markdown("### 💡 Answer")
                            st.info(answer)
                        else:
                            st.warning("No relevant documents found for your question.")
                else:
                    st.warning("Please enter a question!")
            
            # Example questions
            st.markdown("---")
            st.markdown("**Example questions to try:**")
            example_questions = [
                "What is RAG?",
                "How does machine learning work?",
                "What are Large Language Models?",
                "What is the difference between AI and machine learning?",
                "How can RAG help with LLM hallucinations?"
            ]
            
            cols = st.columns(2)
            for i, eq in enumerate(example_questions):
                with cols[i % 2]:
                    if st.button(eq, key=f"example_{i}"):
                        st.session_state.example_question = eq
                        st.rerun()
    
    with tab3:
        st.header("System Information")
        
        st.markdown("""
        ### RAG Architecture Components
        
        #### 1. Document Processing
        - Text chunking for optimal retrieval
        - Overlap between chunks for context continuity
        
        #### 2. Embedding & Indexing
        - Convert text to vector representations
        - Store in vector database for efficient search
        
        #### 3. Retrieval
        - Semantic search using cosine similarity
        - Retrieve top-k most relevant chunks
        
        #### 4. Generation
        - LLM processes query + retrieved context
        - Generates contextual, grounded responses
        
        ### Production Enhancements
        
        To deploy this in production, consider:
        
        1. **Better Embeddings**: Use sentence-transformers or OpenAI embeddings
        2. **LLM Integration**: Connect to OpenAI, Anthropic, or local LLMs
        3. **Vector Database**: Use Pinecone, Weaviate, or ChromaDB
        4. **Caching**: Implement caching for faster responses
        5. **Monitoring**: Track performance and quality metrics
        """)
        
        if st.session_state.rag_system.documents:
            st.markdown("---")
            st.markdown("### Current Knowledge Base")
            st.metric("Total Chunks", len(st.session_state.rag_system.documents))
            
            # Show sample of documents
            with st.expander("Preview Document Chunks"):
                for i, doc in enumerate(st.session_state.rag_system.documents[:5], 1):
                    st.text(f"Chunk {i}: {doc[:200]}...")
                    st.markdown("---")


if __name__ == "__main__":
    main()
