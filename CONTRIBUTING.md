# Contributing to AI Chatbot with RAG

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ChatBot.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add: your feature description"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all public classes and methods
- Keep functions **focused and small** (ideally < 50 lines)
- Use **meaningful variable names**

Example:

```python
def process_document(file_path: Path, chunk_size: int = 500) -> List[str]:
    """
    Process a document into chunks.
    
    Args:
        file_path: Path to the document file
        chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of document chunks
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    pass
```

### Architecture Principles

Follow **SOLID principles**:

- **Single Responsibility**: One class, one purpose
- **Open-Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Implementations should honor interface contracts
- **Interface Segregation**: Small, focused interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

### Project Structure

```
src/
├── interfaces/     # Abstract interfaces only
├── models/         # Data classes and models
├── config/         # Configuration management
├── services/       # Business logic implementations
├── rag/           # RAG-specific components
├── ui/            # User interface implementations
└── exceptions/    # Custom exception classes
```

## Testing

### Running Tests

```powershell
# Test API connections
python tests/test_api.py

# Add unit tests (future)
pytest tests/
```

### Writing Tests

- Create test files in `tests/` directory
- Use descriptive test names: `test_feature_does_something`
- Test edge cases and error conditions
- Mock external dependencies

## Documentation

- Update **README.md** for user-facing changes
- Add **docstrings** for new functions/classes
- Update **type hints** when changing signatures
- Comment **complex logic** (but prefer clear code over comments)

## Commit Messages

Use clear, descriptive commit messages:

```
Add: New feature description
Fix: Bug fix description
Update: Changes to existing feature
Refactor: Code restructuring
Docs: Documentation updates
Test: Adding or updating tests
```

## Pull Request Process

1. **Ensure your code follows the style guide**
2. **Update documentation** as needed
3. **Add tests** for new features
4. **Ensure all tests pass**
5. **Describe your changes** in the PR description
6. **Link related issues** if applicable

## Feature Requests

Have an idea? We'd love to hear it!

1. Check existing issues first
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Potential implementation approach

## Bug Reports

Found a bug? Please report it!

1. Check existing issues first
2. Create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages or logs

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Assume good intentions

## Areas for Contribution

Looking for ideas? Here are areas that need help:

### High Priority
- Unit tests for core components
- Web UI implementation
- Additional document format support
- Performance optimizations

### Medium Priority
- Alternative AI providers (OpenAI, Anthropic)
- Advanced RAG features (re-ranking, hybrid search)
- Conversation export/import
- Multi-user session support

### Documentation
- Tutorial videos
- Use case examples
- Deployment guides
- API documentation

## Development Setup

```powershell
# Clone the repo
git clone https://github.com/your-username/ChatBot.git
cd ChatBot

# Create virtual environment
python -m venv env
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env
# Edit .env with your API key

# Test setup
python tests/test_api.py

# Run the application
python main.py
```

## Questions?

Feel free to:
- Open an issue for discussion
- Reach out to maintainers
- Check existing documentation

Thank you for contributing! 🎉
