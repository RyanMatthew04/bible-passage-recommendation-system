# Scripture Discovery: AI-Powered Bible Search Engine

A semantic search engine that helps users discover relevant Bible passages through intelligent question matching and multi-stage ranking. Built to run efficiently on resource-constrained environments while delivering high-quality results.

## 🎯 Project Overview

This project demonstrates sophisticated problem-solving in building a production-ready semantic search system under severe resource constraints (2 vCPUs, 16GB RAM on DigitalOcean's free tier). The system leverages precomputed embeddings from large models while using smaller models for real-time inference, achieving both accuracy and efficiency.

## ✨ Key Features

- **Semantic Search**: Understands user intent beyond keyword matching
- **Multi-Stage Ranking**: Combines vector similarity with cross-encoder reranking
- **Intelligent Highlighting**: Automatically identifies and highlights the most relevant verses
- **Optimized Performance**: Sub-2-second response times despite resource constraints
- **Beautiful UI**: Modern, responsive interface with gradient backgrounds and smooth animations

## 🏗️ Architecture & Problem-Solving Approach

### The Core Challenge

Running transformer models on a DigitalOcean droplet with only 2 vCPUs and 16GB RAM presents significant challenges:
- **Large models** (Qwen 8B) are too resource-intensive for real-time inference
- **Small models** (Qwen 0.6B) are fast but may lack semantic understanding
- Need to balance **accuracy** with **performance**

### The Solution: Asymmetric Embedding Strategy

I developed a hybrid approach that leverages the strengths of both large and small models:

#### 1. **Offline Precomputation** (Using Jarvis.ai GPU)
- Generated embeddings for ~10,000 biblical questions using **Qwen 8B** model
- Created embeddings for all Bible chapters, verses, and Q&A pairs
- Stored embeddings in efficient FAISS indices for fast retrieval
- **Rationale**: Compute-intensive work done once with powerful hardware

#### 2. **Online Inference** (On DigitalOcean Droplet)
- User queries embedded with lightweight **Qwen 0.6B** model
- Fast similarity search using precomputed embeddings
- **Rationale**: Small model handles real-time workload efficiently

#### 3. **Question Database Strategy**
The system uses questions scraped from [GotQuestions.org](https://gotquestions.org), which claims to have answered **96% of all possible Bible questions**. This comprehensive coverage ensures:
- High probability of finding semantically similar questions for any user query
- Quality control through expert-curated content
- Broad topic coverage across theology, doctrine, and practical faith

## 🔄 Search Pipeline

### Pipeline Flow

```
User Query
    ↓
[1] Embed with Qwen 0.6B
    ↓
[2] FAISS Search: Top 100 Questions (from Qwen 8B Embeddings)
    ↓
[3] Cross-Encoder Rerank (STSB Model) → Top 5 Questions
    ↓
[4] Average Top 5 Question Embeddings (from Qwen 8B)
    ↓
[5] For Each of Top 5 Questions:
    ├─ FAISS Search: Top 20 Q&A Pairs
    └─ Cross-Encoder Rerank (QNLI Model) → Top 2 Q&A per Question
    ↓
[6] Collect All Top Q&A Pairs → Deduplicate
    ↓
[7] Extract Bible References from Q&A Text
    ↓
[8] Filter Chapters from Database
    ↓
[9] Compute Cosine Similarity with Averaged Question Embedding
    ↓
[10] Select Top 10 Chapters
    ↓
[11] Cross-Encoder Rerank (QNLI Model) → Final Chapter Ranking
    ↓
[12] For Each Chapter:
    ├─ Get All Verses
    ├─ Compute Verse Similarities
    ├─ Select Top 10 Verses
    └─ Cross-Encoder Rerank → Top 3 Verses for Highlighting
    ↓
Return Complete Chapters with Highlighted Verses
```

## 📊 Detailed Pipeline Explanation

### Stage 1: Question Matching
**Goal**: Find the most semantically similar questions to the user's query

1. **Embed User Query** (Qwen 0.6B)
   - Lightweight model ensures fast response
   - Normalized embeddings for cosine similarity

2. **FAISS Similarity Search**
   - Search through 10K precomputed Qwen 8B question embeddings
   - Retrieve top 100 candidates
   - **Why 100?** Cast a wide net for recall while keeping computation manageable

3. **Cross-Encoder Reranking** (STSB RoBERTa)
   - More accurate than bi-encoder for final ranking
   - STSB (Semantic Textual Similarity) model specifically trained for measuring question similarity
   - Select top 5 questions
   - **Why cross-encoder?** Attention mechanism considers query-question interactions

4. **Embedding Averaging**
   - Average the Qwen 8B embeddings of top 5 questions
   - Creates a robust query representation
   - **Rationale**: Reduces noise, captures multiple semantic aspects

### Stage 2: Q&A Retrieval
**Goal**: Find relevant question-answer pairs that reference Bible passages

1. **Individual Question Searches**
   - For each of the 5 top questions, search Q&A database
   - Retrieve top 20 Q&A pairs per question
   - **Why separate searches?** Each question may target different aspects

2. **Cross-Encoder Reranking** (QNLI ELECTRA)
   - QNLI (Question-Natural Language Inference) model
   - Trained to determine if text answers a question
   - Select top 2 Q&A pairs per question
   - **Result**: Maximum 10 high-quality Q&A pairs

3. **Deduplication**
   - Remove duplicate Q&A pairs across searches
   - Ensures diverse results

### Stage 3: Reference Extraction & Chapter Retrieval
**Goal**: Identify and rank relevant Bible chapters

1. **Reference Extraction**
   - Parse Q&A text for Bible references (e.g., "John 3:16", "Romans 8")
   - Comprehensive regex matching with book name normalization
   - **Challenge solved**: Handles abbreviations (Gen, Ge, Genesis)

2. **Chapter Filtering**
   - Match extracted references to chapter database
   - Deduplicate while preserving order

3. **Similarity Ranking**
   - Compute cosine similarity between averaged question embedding and chapter embeddings
   - Select top 10 chapters

4. **Cross-Encoder Reranking** (QNLI)
   - Rerank chapters based on relevance to original query
   - Uses full chapter text for context
   - **Final output**: Top 10 most relevant chapters

### Stage 4: Verse-Level Analysis
**Goal**: Provide complete chapters with intelligent verse highlighting

1. **Retrieve All Verses**
   - For each chapter, get complete verse list (sorted)
   - **Design decision**: Show full context, not just snippets

2. **Verse Similarity Scoring**
   - Compute similarity for every verse in chapter
   - Provides fine-grained relevance scores

3. **Top Verse Identification**
   - Select top 10 verses by similarity
   - Apply cross-encoder reranking (QNLI)
   - Identify top 3 verses for highlighting

4. **Highlighting Logic**
   - Gradient highlighting based on reranking scores
   - Highest scored verse gets strongest highlight
   - **UX benefit**: Users immediately see most relevant content

5. **Final Scoring**
   - Combine chapter similarity with top 3 verse similarities
   - Mean score determines final chapter ranking
   - Ensures chapters with highly relevant verses rise to the top

## 🧠 Model Selection Rationale

| Model | Purpose | Why This Model? |
|-------|---------|----------------|
| **Qwen 0.6B** | Query embedding | Fast inference, good semantic understanding for size, low memory footprint |
| **Qwen 8B** | Precomputed embeddings | Superior semantic representation, captures nuanced meanings |
| **STSB RoBERTa** | Question similarity | Specifically trained on semantic textual similarity, excellent for matching questions |
| **QNLI ELECTRA** | Answer relevance | Trained on natural language inference, determines if text answers question |

## 🚀 Performance Optimizations

### 1. **FAISS Index Strategy**
```python
faiss.omp_set_num_threads(os.cpu_count())  # Maximize parallelism
```
- Inner product search on normalized embeddings (= cosine similarity)
- In-memory indices for sub-millisecond lookups

### 2. **Batch Processing**
- Cross-encoder predictions batched for efficiency
- Parallel embedding generation offline

### 3. **Smart Caching**
- Singleton pattern for model loading
- Indices loaded once, reused across requests

### 4. **Progressive Results**
- Multi-stage pipeline allows early termination if needed
- Each stage filters aggressively to reduce downstream computation

## 📁 Project Structure

```
bible_search/
├── views.py              # Django API endpoints
├── utils.py              # Search engine implementation
├── urls.py               # URL routing
├── templates/
│   └── bible.html        # Frontend interface
├── vector_db/            # Precomputed embeddings & indices
│   ├── question_index_qwen3-embeddin_0_6b.faiss
│   ├── question_embeddings_qwen3_embedding_8b.npy
│   ├── qna_index_qwen3_embedding_8b.faiss
│   ├── chapters_index_qwen3_embedding_8b.faiss
│   └── verses_index_qwen3_embedding_8b.faiss
└── README.md
```

## 🛠️ Tech Stack

- **Backend**: Django 4.0+
- **Embedding Models**: 
  - Qwen/Qwen3-Embedding-0.6B (runtime)
  - Qwen/Qwen3-Embedding-8B (precomputed)
- **Reranking Models**:
  - cross-encoder/stsb-roberta-large
  - cross-encoder/qnli-electra-base
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Deployment**: DigitalOcean Droplet (2 vCPUs, 16GB RAM)
- **GPU Compute**: Jarvis.ai (for precomputation)


## 🙏 Acknowledgments

- **GotQuestions.org** for the comprehensive question database
- **Jarvis.ai** for GPU resources for embedding precomputation
- **Qwen Team** for the excellent embedding models
- **Hugging Face** for the cross-encoder models
- **FAISS Team** for the efficient similarity search library

---

**Note**: This project demonstrates how careful architectural decisions and creative problem-solving can overcome resource constraints to deliver a production-quality application. The asymmetric embedding strategy and multi-stage ranking pipeline are techniques applicable to many semantic search problems beyond Bible search.
