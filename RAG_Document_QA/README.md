### RAG (Retrieval-Augmented Generation) Document Q&A System

A modern demonstration of RAG architecture for question-answering over documents.

#### Overview
This application showcases the RAG (Retrieval-Augmented Generation) pattern, which combines information retrieval with language generation to provide accurate, grounded answers to questions about your documents.

#### What is RAG?
RAG is a technique that enhances language models by:
1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the query with this context
3. **Generating** accurate responses based on retrieved facts

This approach reduces hallucinations and provides more accurate, source-grounded responses.

#### Features

##### Core Functionality
- 📄 **Document Upload**: Text files, manual input, or sample documents
- ✂️ **Smart Chunking**: Automatically splits documents into optimal chunks
- 🔍 **Semantic Search**: Find relevant information using similarity search
- 💬 **Contextual Q&A**: Generate answers based on retrieved context
- 📊 **Transparency**: View retrieved chunks and relevance scores

##### User Interface
- **3-Tab Design**: Upload, Query, System Info
- **Sample Data**: Pre-loaded AI/ML content for immediate testing
- **Adjustable Retrieval**: Configure number of chunks to retrieve
- **Context Display**: See exactly what information was used

#### Architecture

```
User Query → Embedding → Similarity Search → Top-K Chunks → Context + Query → Answer Generation
                ↑                              ↑
         Document Store              Vector Embeddings
```

#### How It Works

1. **Document Processing**
   - Upload documents (text files or manual input)
   - System splits text into manageable chunks (~500 words)
   - Chunks are stored in memory

2. **Question Processing**
   - User asks a question
   - System creates embedding of the question
   - Searches for most similar document chunks

3. **Answer Generation**
   - Top-K most relevant chunks are retrieved
   - System generates answer using retrieved context
   - Shows source chunks and confidence scores

#### Requirements
```bash
streamlit>=1.39.0
numpy>=1.26.0
```

#### Installation & Running

```bash
# Navigate to directory
cd RAG_Document_QA

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Usage Tips

1. **Start with Sample Data**: Check "Use sample documents" to explore with pre-loaded content
2. **Try Example Questions**: Use the suggested questions to understand capabilities
3. **View Retrieved Context**: Expand the "Retrieved Context" section to see source material
4. **Adjust Retrieval**: Use the sidebar slider to control how many chunks to retrieve

#### Example Questions

With the sample data, try:
- "What is RAG?"
- "How does machine learning work?"
- "What are Large Language Models?"
- "How can RAG help with hallucinations?"

#### Demo vs Production

This is a **simplified demonstration** for learning purposes.

| Feature | Demo | Production |
|---------|------|------------|
| Embeddings | Simple word-based | Sentence-transformers, OpenAI |
| Vector Store | In-memory list | Pinecone, Weaviate, ChromaDB |
| LLM | Simulated | GPT-4, Claude, Llama |
| Scalability | Small documents | Millions of documents |
| Persistence | Session only | Database-backed |

#### Production Enhancement Guide

##### 1. Advanced Embeddings

```python
# Using sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```

```python
# Using OpenAI
import openai

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=texts
)
embeddings = [item['embedding'] for item in response['data']]
```

##### 2. Vector Database

```python
# Using Pinecone
import pinecone

pinecone.init(api_key="your-key")
index = pinecone.Index("document-qa")
index.upsert(vectors=embeddings, metadata=metadata)
results = index.query(query_embedding, top_k=5)
```

```python
# Using ChromaDB
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")
collection.add(documents=texts, embeddings=embeddings)
results = collection.query(query_texts=["question"], n_results=5)
```

##### 3. LLM Integration

```python
# Using OpenAI
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Answer based on context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
)
answer = response.choices[0].message.content
```

```python
# Using Anthropic Claude
import anthropic

client = anthropic.Client(api_key="your-key")
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{
        "role": "user",
        "content": f"Context: {context}\n\nQuestion: {question}"
    }]
)
answer = response.content[0].text
```

#### Key Concepts

- **Chunking Strategy**: Breaking documents into optimal sizes for retrieval
- **Semantic Search**: Finding relevant information by meaning, not just keywords
- **Context Window**: Amount of information to provide to the LLM
- **Grounding**: Ensuring answers are based on actual document content
- **Retrieval Quality**: Balance between too much and too little context

#### Use Cases

1. **Internal Knowledge Base**: Company documentation Q&A
2. **Customer Support**: Automated responses from help docs
3. **Research Assistant**: Query academic papers and research
4. **Legal/Compliance**: Search policy documents
5. **Education**: Interactive learning from textbooks

#### Troubleshooting

**Q: No relevant documents found?**
- A: Try rephrasing your question or adjust the retrieval count

**Q: Answers seem generic?**
- A: This demo uses simple extraction. Production systems use advanced LLMs

**Q: How to handle large documents?**
- A: Upload in sections or use file splitting utilities before upload

#### Performance Optimization

- **Caching**: Store embeddings to avoid recomputation
- **Batch Processing**: Process multiple documents simultaneously
- **Async Queries**: Handle multiple questions concurrently
- **Compression**: Use efficient embedding models

#### Next Steps

1. **Experiment** with different chunk sizes
2. **Try** your own documents
3. **Learn** about vector databases
4. **Explore** LLM APIs (OpenAI, Anthropic)
5. **Build** a production RAG system

#### Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Vector Database](https://www.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)

#### Credits

**Created by**: Sazzad Hussain Farhaan  
**Email**: shfarhaan21@gmail.com  
**Last Updated**: January 2026

---
*Modern AI application demonstrating RAG architecture with Streamlit v1.39.0+*
