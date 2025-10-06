# Quick Reference Guide

## Common Commands

### Setup
```powershell
# Initial setup
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your API key
```

### Testing
```powershell
# Validate API and embeddings
python tests/test_api.py
```

### Document Management
```powershell
# Ingest new documents
python scripts/ingest_documents.py

# Add documents to this folder:
data/documents/
```

### Running
```powershell
# Start with Web UI (Recommended) - Easy way
# Double-click: start_web_ui.bat

# Or use full path:
.\env\Scripts\streamlit.exe run app.py

# Or start with CLI
python main.py
```

## File Locations

| What | Where |
|------|-------|
| Web UI launcher | `start_web_ui.bat` / `.ps1` |
| Web UI application | `app.py` |
| CLI application | `main.py` |
| Configuration | `.env` |
| Documents to ingest | `data/documents/` |
| Tests | `tests/` |
| Utility scripts | `scripts/` |
| Source code | `src/` |
| Logs | `*.log` (auto-generated) |
| Vector database | `chroma_db/` (auto-generated) |

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Chatbot Engine | `src/services/chatbot.py` | Main orchestration |
| AI Provider | `src/services/groq_provider.py` | Groq API integration |
| Embeddings | `src/rag/minilm_embeddings.py` | Local embeddings |
| Vector Store | `src/rag/vector_store.py` | ChromaDB integration |
| CLI Interface | `src/ui/cli_interface.py` | User interaction |
| Configuration | `src/config/settings.py` | Settings management |

## Environment Variables

```env
# Required
GROQ_API_KEY=gsk_your_key_here

# Optional (defaults shown)
GROQ_MODEL=llama-3.1-8b-instant
MAX_TOKENS=1024
TEMPERATURE=0.7
MAX_HISTORY_LENGTH=20
```

## Chatbot Commands

While using the chatbot:

- `/help` - Show help
- `/clear` - Clear conversation
- `/summary` - Show statistics
- `/quit` - Exit

## Troubleshooting

### No documents found
```powershell
# Add files to data/documents/
# Then run:
python scripts/ingest_documents.py
```

### API key error
```powershell
# Check .env file exists and contains:
GROQ_API_KEY=gsk_your_actual_key
```

### Import errors
```powershell
# Ensure virtual environment is activated:
.\env\Scripts\Activate.ps1

# Reinstall dependencies:
pip install -r requirements.txt
```

### Model download issues
```powershell
# Clear cache and retry:
Remove-Item -Recurse -Force ~/.cache/torch/sentence_transformers/
python tests/test_api.py
```

## Development Workflow

1. **Activate environment**
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

2. **Make changes**
   - Edit files in `src/`
   - Follow SOLID principles
   - Add docstrings

3. **Test changes**
   ```powershell
   python tests/test_api.py
   python main.py
   ```

4. **Commit**
   ```powershell
   git add .
   git commit -m "Add: feature description"
   ```

## Adding New Features

### New AI Provider
1. Create file in `src/services/`
2. Implement `IAIProvider` interface
3. Update `main.py` to use new provider

### New Document Format
1. Update `scripts/ingest_documents.py`
2. Add processing logic for new format
3. Test with sample files

### New UI
1. Create file in `src/ui/`
2. Implement `IUserInterface` interface
3. Update `main.py` to use new UI

## Useful Links

- [Groq Console](https://console.groq.com/) - Get API keys
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [ChromaDB Docs](https://docs.trychroma.com/) - Vector database
- [Python Dotenv](https://pypi.org/project/python-dotenv/) - Environment variables

## Support

- Check `README.md` for detailed documentation
- Read `CONTRIBUTING.md` for development guidelines
- See `PROJECT_ORGANIZATION.md` for architecture details
- Open an issue for bugs or questions
