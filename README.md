# 🤖 Agentic RAG Chatbot with Plan → Act → Reflect

A **state-of-the-art chatbot** featuring **Agentic RAG (Retrieval-Augmented Generation)** with sophisticated **Plan → Act → Reflect** reasoning cycles. This production-ready application implements industry best practices for enterprise-grade document-based AI systems.

## 🚀 Key Features

### 🧠 Agentic Intelligence
- 🔄 **Plan → Act → Reflect Cycles**: Multi-step reasoning with self-correction
- 🧩 **Query Decomposition**: Breaks complex queries into manageable parts  
- 🎯 **Intent Routing**: Intelligent query classification and routing
- ⚖️ **LLM-as-a-Judge**: Evaluates retrieval quality and adapts strategy
- 🔄 **Reflection Loops**: Refines queries when confidence is low

### 🔍 Advanced Retrieval
- � **Hybrid Search**: Combines semantic vector search with BM25 keyword search
- 🎯 **Cross-Encoder Re-ranking**: MS-MARCO model for precise relevance scoring
- 📊 **Confidence-Based Decisions**: Industry-standard thresholds (0.75/0.5)
- 📝 **Source Attribution**: Every claim linked to specific documents

### 🎨 Adaptive Generation  
- 🌡️ **Dynamic Temperature**: 0.1 (factual) to 0.8 (general) based on confidence
- 📋 **Strategy Selection**: Factual/Hybrid/General response modes
- 🛡️ **Guardrails**: Prevents hallucination with strict document adherence
- � **Verifiable Citations**: Format: [Source X: document_name]

### 🏗️ Enterprise Architecture
- � **Groq AI Integration**: Powered by fast LLaMA 3.1 models
- 📚 **ChromaDB Vector Store**: Persistent document embeddings
- 🧮 **Local MiniLM Embeddings**: No external API calls for embeddings
- 🌐 **Modern Streamlit UI**: Real-time chat with reasoning display
- 🖥️ **CLI Interface**: Command-line option for developers
- 🔧 **SOLID Principles**: Clean, maintainable, extensible codebase

## Architecture

The application follows SOLID principles:

- **Single Responsibility Principle (SRP)**: Each class has one reason to change
- **Open-Closed Principle (OCP)**: Open for extension, closed for modification
- **Liskov Substitution Principle (LSP)**: Interfaces can be replaced with implementations
- **Interface Segregation Principle (ISP)**: Clients depend only on methods they use
- **Dependency Inversion Principle (DIP)**: Depend on abstractions, not concretions

### Project Structure

```
ChatBot/
├── src/
│   ├── interfaces/              # Abstract interfaces (SOLID compliance)
│   │   └── core.py             # Core interfaces for all components
│   ├── models/                 # Data models
│   │   └── message.py          # Message data structure
│   ├── config/                 # Configuration management
│   │   └── settings.py         # Environment-based configuration
│   ├── services/               # Business logic implementations
│   │   ├── groq_provider.py    # Groq AI service implementation
│   │   ├── message_handler.py  # Conversation management
│   │   └── chatbot.py          # Main chatbot engine with RAG
│   ├── rag/                    # Retrieval-Augmented Generation system
│   │   ├── minilm_embeddings.py    # Local embedding service
│   │   ├── vector_store.py         # ChromaDB integration
│   │   ├── document_processor.py   # Document text processing
│   │   ├── pipeline.py             # RAG pipeline orchestration
│   │   └── state_manager.py        # RAG state management
│   ├── ui/                     # User interface implementations
│   │   └── cli_interface.py    # Command line interface
│   └── exceptions/             # Custom exception classes
│       └── chatbot_exceptions.py
├── tests/                      # Test files
│   └── test_api.py            # API validation tests
├── scripts/                    # Utility scripts
│   └── ingest_documents.py    # Document ingestion script
├── data/
│   └── documents/             # Place your .txt/.pdf files here
├── chroma_db/                 # Vector database (auto-generated)
├── main.py                    # Application entry point
├── .env                       # Environment variables (create from .env.example)
├── .gitignore                 # Git ignore patterns
└── requirements.txt           # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8 or higher
- Groq API key (get one from [Groq Console](https://console.groq.com/))
- Approximately 500MB disk space for ML models

### Installation

1. **Clone or download this project**

2. **Set up virtual environment** (recommended):
   ```powershell
   # Windows PowerShell
   python -m venv env
   .\env\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `groq` - AI chat completions
   - `sentence-transformers` - Local embeddings
   - `chromadb` - Vector database
   - `torch` - ML framework for embeddings
   - `pdfplumber` - PDF document processing

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   # Required
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   
   # Optional (defaults shown)
   GROQ_MODEL=llama-3.1-8b-instant
   MAX_TOKENS=1024
   TEMPERATURE=0.7
   MAX_HISTORY_LENGTH=20
   SYSTEM_PROMPT=You are a helpful and friendly AI assistant.
   ```

5. **Verify setup**:
   ```powershell
   python tests/test_api.py
   ```
   This will validate your Groq API key and test the embedding system.

### Initial Document Ingestion

To enable RAG (document-based answers), ingest your documents:

1. **Add documents** to the `data/documents/` folder:
   - Supported formats: `.txt`, `.pdf`
   - Example: employee_benefits.txt, policies.pdf, etc.

2. **Run the ingestion script**:
   ```powershell
   python scripts/ingest_documents.py
   ```

3. **Verify ingestion**:
   The script will show how many chunks were added to the database.

### Running the Chatbot

**Option 1: Web UI (Recommended)**

Easiest way:
```powershell
# Double-click start_web_ui.bat (or start_web_ui.ps1)
# Or run manually:
.\env\Scripts\streamlit.exe run app.py
```
This opens a modern web interface at http://localhost:8501

**Option 2: Command Line**
```powershell
python main.py
```

On startup, you'll see:
- Document retrieval system status
- Number of indexed document chunks
- Available commands

## Usage

### Web UI (Streamlit)

The web interface provides:
- 💬 **Chat Interface**: Modern chat UI with message history
- 📊 **Sidebar**: Document status, statistics, and controls
- 🎨 **Rich Formatting**: Markdown support for better readability
- 🔄 **Real-time Updates**: Instant responses with typing indicators

Simply run `streamlit run app.py` and open your browser.

### Basic Chat (CLI)

Just type your message and press Enter:

```
You: Hello, how are you?
Assistant: Hello! I'm doing well, thank you for asking. How can I help you today?
```

### Document-Based Questions (RAG)

If you've ingested documents, the chatbot will automatically retrieve relevant information:

```
You: What are the employee health benefits?
Assistant: According to our Employee Benefits Guide, employees get:
- Standard health insurance with major medical coverage
- Prescription medication coverage
- Doctor visits and hospital stays
...
```

The chatbot will:
1. Search the vector database for relevant document chunks
2. Use only the retrieved documents to answer
3. Cite specific sources when available

### Available Commands

- `/help` - Show help message and available commands
- `/clear` - Clear conversation history (keeps documents)
- `/summary` - Show conversation statistics
- `/quit` or `/exit` - Exit the chatbot

### Configuration Options

Customize behavior through `.env` file:

```env
# AI Provider
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant    # Groq model to use
MAX_TOKENS=1024                     # Max response length
TEMPERATURE=0.7                     # Creativity (0.0-1.0)

# Conversation
SYSTEM_PROMPT=You are a helpful AI assistant.
MAX_HISTORY_LENGTH=20               # Messages to keep in context

# Embeddings (defaults work well)
EMBEDDING_PROVIDER=minilm
MINILM_MODEL_NAME=all-MiniLM-L6-v2
```

## Architecture & Design

### SOLID Principles

This project demonstrates clean architecture:

- **Single Responsibility**: Each class has one clear purpose
- **Open-Closed**: Open for extension (new providers/UIs), closed for modification
- **Liskov Substitution**: All interfaces can be swapped with implementations
- **Interface Segregation**: Focused interfaces (IAIProvider, IMessageHandler, etc.)
- **Dependency Inversion**: Depends on abstractions, not concrete implementations

### Component Overview

#### Core Services
- **ChatbotEngine** (`chatbot.py`): Orchestrates all components
- **GroqProvider** (`groq_provider.py`): Handles AI completions
- **MessageHandler** (`message_handler.py`): Manages conversation history
- **CLIInterface** (`cli_interface.py`): User interaction layer

#### RAG System
- **MiniLMEmbeddingService**: Generates 384-dim embeddings locally
- **ChromaVectorStore**: Stores and searches document vectors
- **DocumentProcessor**: Chunks and processes documents
- **RAGPipeline**: Orchestrates retrieval and generation

#### Configuration
- Environment-based settings
- Type-safe configuration access
- Validation on startup

### Data Flow

```
User Input → CLI Interface → Chatbot Engine
                               ↓
                    RAG Search (if enabled)
                               ↓
                    Vector Store Query
                               ↓
                    Document Retrieval
                               ↓
                    Enhanced Prompt Creation
                               ↓
                    Groq AI Provider
                               ↓
                    Response → User
```

## Extending the Application

### Adding a New AI Provider

```python
from src.interfaces.core import IAIProvider, IConfiguration
from typing import List
from src.models.message import Message

class OpenAIProvider(IAIProvider):
    def __init__(self, config: IConfiguration):
        self.api_key = config.get_api_key()
        # Initialize OpenAI client
    
    async def generate_response(self, messages: List[Message]) -> str:
        # Implement OpenAI API call
        pass
    
    def is_available(self) -> bool:
        return bool(self.api_key)
```

### Adding a Web UI

```python
from src.interfaces.core import IUserInterface, IChatbot
from fastapi import FastAPI

class WebInterface(IUserInterface):
    def __init__(self, chatbot: IChatbot):
        self.chatbot = chatbot
        self.app = FastAPI()
        self._setup_routes()
    
    async def start(self):
        # Start FastAPI server
        pass
```

### Adding New Document Types

Extend `DocumentProcessor` to handle new formats:

```python
def process_docx(self, file_path: Path) -> List[str]:
    from docx import Document
    doc = Document(file_path)
    return [p.text for p in doc.paragraphs if p.text.strip()]
```

## Development

### Running Tests

Test the API connections:
```powershell
python tests/test_api.py
```

This validates:
- Groq API authentication and connectivity
- MiniLM embedding generation
- Model downloads and initialization

### Adding Unit Tests

Create test files in the `tests/` directory:

```python
# tests/test_chatbot.py
import pytest
from src.services.chatbot import SimpleChatbot

@pytest.mark.asyncio
async def test_chatbot_initialization():
    # Your test code
    pass
```

### Code Quality

- Follow PEP 8 style guide
- Use type hints for function signatures
- Write docstrings for public methods
- Keep functions focused and small

### Project Scripts

- `scripts/ingest_documents.py` - Ingest documents into vector DB
- `tests/test_api.py` - Validate API setup

## Troubleshooting

### Common Issues

**"GROQ_API_KEY environment variable is required"**
- Create `.env` file in project root
- Ensure key starts with `gsk_`
- Get a new key from https://console.groq.com/keys

**"sentence-transformers not found"**
- Run: `pip install sentence-transformers torch`
- First run downloads ~80MB model (cached afterward)

**"No documents found" warning**
- Add `.txt` or `.pdf` files to `data/documents/`
- Run: `python scripts/ingest_documents.py`
- Check ingestion logs for errors

**RAG not retrieving relevant chunks**
- Check similarity threshold in `chatbot.py` (default: 0.1)
- Ensure documents are properly formatted
- Verify ChromaDB collection exists: `chroma_db/` folder

**Import errors**
- Ensure virtual environment is activated
- Install all dependencies: `pip install -r requirements.txt`
- Run from project root directory

### Logging

Logs are written to `chatbot.log`:

```python
# View recent logs
tail -f chatbot.log  # Linux/Mac
Get-Content chatbot.log -Tail 20  # PowerShell
```

Log levels:
- `INFO` - Normal operations
- `WARNING` - Recoverable issues
- `ERROR` - Failures requiring attention

### Performance Tips

1. **First run is slow**: Model downloads take time but are cached
2. **Large documents**: Chunk size affects memory usage (default: paragraphs)
3. **Many documents**: Consider increasing `max_results` in RAG search
4. **Slow responses**: Check Groq API rate limits and model choice

## Deployment

### Production Considerations

1. **Environment Variables**: Use secrets management (Azure Key Vault, AWS Secrets Manager)
2. **Database**: Consider hosted ChromaDB or Pinecone for scale
3. **Monitoring**: Add application insights and error tracking
4. **Rate Limiting**: Implement request throttling
5. **Caching**: Cache frequent queries to reduce API calls

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow SOLID principles and existing code style
4. Add tests for new features
5. Update documentation
6. Submit a pull request

### Contribution Guidelines

- Follow PEP 8 style guide
- Maintain test coverage
- Update README for new features
- Keep commits focused and descriptive

## Roadmap

Potential future enhancements:

- [ ] Web UI with FastAPI/Flask
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Additional AI providers (OpenAI, Anthropic)
- [ ] Advanced RAG with re-ranking
- [ ] Conversation export/import
- [ ] Multi-user support with sessions
- [ ] Integration with document management systems

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Groq](https://groq.com/) - Ultra-fast AI inference
- [Sentence Transformers](https://www.sbert.net/) - State-of-the-art embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://langchain.com/) - Inspiration for RAG patterns