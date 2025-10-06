# Agentic RAG Chatbot - Codebase Organization

## 🏗️ Architecture Overview

This codebase implements a sophisticated **Agentic RAG (Retrieval-Augmented Generation) System** using the **Plan → Act → Reflect** paradigm for intelligent document-based question answering.

## 📁 Project Structure

```
ChatBot/
├── 📱 Entry Points
│   ├── app.py                 # Streamlit web interface (Agentic RAG UI)
│   ├── main.py               # CLI interface (Agentic RAG CLI)
│   ├── start_web_ui.bat     # Windows launcher script
│   └── start_web_ui.ps1     # PowerShell launcher script
│
├── 📊 Data & Storage
│   ├── data/documents/       # Source documents for RAG
│   ├── chroma_db/           # Vector database (ChromaDB)
│   └── requirements.txt     # Python dependencies
│
├── 🔧 Scripts
│   └── ingest_documents.py  # Document processing & indexing
│
├── 🧠 Core System (src/)
│   ├── 🎮 Services
│   │   ├── chatbot.py           # 🤖 AgenticChatbot (main orchestrator)
│   │   ├── agentic_rag.py       # 🚀 AgenticRAGAgent (Plan→Act→Reflect)
│   │   ├── groq_provider.py     # 🌐 Groq AI API provider
│   │   └── message_handler.py   # 💬 Conversation management
│   │
│   ├── 🔍 RAG Components
│   │   ├── vector_store.py      # 📚 Enhanced vector storage (ChromaDB)
│   │   ├── hybrid_search.py     # 🔍 Hybrid search (Semantic + BM25)
│   │   ├── reranker.py         # 🎯 Cross-encoder re-ranking
│   │   ├── document_processor.py # 📄 Recursive text splitting
│   │   └── minilm_embeddings.py # 🧮 Local sentence embeddings
│   │
│   ├── ⚙️ Configuration
│   │   └── settings.py         # 🔧 App configuration & environment
│   │
│   ├── 🎯 Interfaces
│   │   └── core.py            # 📋 Abstract interfaces & contracts
│   │
│   ├── 📨 Models
│   │   └── message.py         # 💬 Message data structures
│   │
│   ├── ⚠️ Exceptions
│   │   └── chatbot_exceptions.py # 🚨 Error handling
│   │
│   └── 🖥️ UI Components
│       ├── cli_interface.py    # 💻 Command-line interface
│       └── streamlit_interface.py # 🌐 Web UI components
│
└── 📝 Documentation
    ├── README.md              # Project overview
    ├── QUICK_REFERENCE.md     # Quick commands & usage
    ├── PROJECT_ORGANIZATION.md # Architecture details
    └── CONTRIBUTING.md        # Development guidelines
```

## 🚀 Agentic RAG Architecture

### Core Components

#### 1. 🤖 AgenticChatbot (`src/services/chatbot.py`)
- **Primary orchestrator** integrating all agentic components
- Manages conversation flow and user interactions
- Provides adaptive response generation based on confidence

#### 2. 🧠 AgenticRAGAgent (`src/services/agentic_rag.py`)
- **Plan → Act → Reflect reasoning engine**
- Implements 6-stage processing pipeline:
  1. **Query Analysis** - Decomposition & intent routing
  2. **Intelligent Retrieval** - Hybrid search execution
  3. **Confidence Assessment** - LLM-as-a-Judge evaluation
  4. **Reflection** - Query refinement & retry logic
  5. **Final Generation** - Adaptive response synthesis
  6. **Source Attribution** - Verifiable citations

#### 3. 🔍 Hybrid Search System (`src/rag/hybrid_search.py`)
- **Semantic Search**: Vector similarity using MiniLM embeddings
- **Keyword Search**: BM25 scoring for exact term matching
- **Reciprocal Rank Fusion**: Combines rankings optimally
- **Adaptive Weighting**: Query-dependent score balancing

#### 4. 🎯 Cross-Encoder Re-ranker (`src/rag/reranker.py`)
- **Precision Re-ranking**: MS-MARCO cross-encoder model
- **Top-K Selection**: Refines top 50 to definitive top 5
- **Confidence Scoring**: Very High/High/Medium/Low levels
- **Fallback Strategy**: Heuristic re-ranking when model unavailable

#### 5. 📚 Enhanced Vector Store (`src/rag/vector_store.py`)
- **Industry Thresholds**: High (≥0.75), Low (≥0.5) confidence
- **Confidence Evaluation**: Multi-dimensional assessment
- **Retrieval Strategy**: Adaptive response generation
- **ChromaDB Integration**: Persistent vector storage

## 🎯 Key Features

### ✨ Agentic Capabilities
- **🔄 Plan → Act → Reflect Cycles**: Multi-step reasoning with self-correction
- **🧩 Query Decomposition**: Breaks complex queries into manageable parts
- **🎯 Intent Routing**: Classifies and routes queries appropriately
- **⚖️ LLM-as-a-Judge**: Evaluates retrieval quality and adapts strategy
- **🔄 Reflection Loops**: Refines queries when confidence is low

### 🔍 Advanced Retrieval
- **🌊 Hybrid Search**: Semantic + keyword search for maximum recall
- **🎯 Cross-Encoder Re-ranking**: Precise relevance scoring
- **📊 Confidence-Based Decisions**: Adaptive thresholds and strategies
- **📝 Source Attribution**: Every claim linked to specific documents

### 🎨 Adaptive Generation
- **🌡️ Dynamic Temperature**: 0.1 (factual) to 0.8 (general)
- **📋 Strategy Selection**: Factual/Hybrid/General based on confidence
- **🔗 Source Citations**: Format: [Source X: document_name]
- **🛡️ Guardrails**: Prevents hallucination with strict document adherence

## 🚀 Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
# Visit: http://localhost:8501
```

### CLI Interface
```bash
python main.py
```

### Document Ingestion
```bash
python scripts/ingest_documents.py
```

## 🔧 Configuration

- **Environment**: `.env` file with `GROQ_API_KEY`
- **Documents**: Place files in `data/documents/`
- **Vector DB**: Stored in `chroma_db/` directory
- **Logging**: Outputs to `chatbot.log`

## 📊 Monitoring & Debugging

The system provides comprehensive logging and reasoning transparency:

- **🔍 Agent Reasoning**: Step-by-step thought process display
- **📊 Confidence Metrics**: Real-time confidence scoring
- **🎯 Search Statistics**: Retrieval and re-ranking performance
- **📈 Component Status**: Health monitoring for all subsystems

## 🎯 Benefits of This Architecture

1. **🧠 Intelligence**: Plan→Act→Reflect enables sophisticated reasoning
2. **🎯 Precision**: Cross-encoder re-ranking ensures relevant results
3. **🔍 Recall**: Hybrid search maximizes document coverage
4. **⚖️ Reliability**: Confidence assessment prevents hallucination
5. **📝 Transparency**: Full reasoning and source attribution
6. **🔄 Adaptability**: Self-correcting through reflection cycles
7. **🚀 Performance**: Optimized for both accuracy and speed

This represents the **state-of-the-art in RAG architecture**, implementing industry best practices for enterprise-grade document-based AI systems.