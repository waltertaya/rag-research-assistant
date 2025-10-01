# RAG Research Assistant

A powerful Retrieval-Augmented Generation (RAG) system for document analysis and question answering. This tool allows you to ingest various document formats, create vector embeddings, and perform intelligent queries using OpenAI's language models.

## Features

- **Multi-format Document Support**: PDF, DOCX, TXT, Markdown, and CSV files
- **Intelligent Text Chunking**: Token-based text splitting for optimal embedding quality
- **Vector Search**: FAISS-powered similarity search for relevant document retrieval
- **OpenAI Integration**: GPT-4 and embedding models for accurate responses
- **Dual Interface**: Both CLI and Streamlit web interface
- **Embedding Caching**: Reduces API costs by caching generated embeddings
- **Metadata Tracking**: Preserves document source information and chunk references

## Architecture

The system consists of several key components:

- **Document Ingestion**: Parses and extracts text from various file formats
- **Text Chunking**: Splits documents into manageable chunks for embedding
- **Vector Indexing**: Creates and manages FAISS vector indices
- **Retrieval System**: Finds relevant document chunks based on semantic similarity
- **Response Generation**: Uses retrieved context to generate accurate answers

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/waltertaya/rag-research-assistant.git
cd rag-research-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
DATA_DIR=data
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
INDEX_DIR=data/index
```

## Usage

### Command Line Interface (CLI)

#### Ingest Documents
Add documents to your knowledge base:
```bash
python -m src.cli.cli ingest path/to/your/document.pdf
```

#### Query Documents
Ask questions about your ingested documents:
```bash
python -m src.cli.cli query "What is the main topic of this document?"
```

With custom retrieval settings:
```bash
python -m src.cli.cli query "Explain the methodology" --top-k 10
```

### Web Interface (Streamlit)

Launch the interactive web interface:
```bash
streamlit run src/ui/app.py
```

Then open your browser to `http://localhost:8501` to:
1. Upload documents through the web interface
2. Ingest them into the vector database
3. Ask questions and get AI-powered answers with source citations

## Supported File Formats

- **PDF**: `.pdf` - Extracted using pdfplumber
- **Word Documents**: `.docx` - Parsed using python-docx
- **Text Files**: `.txt`, `.md`, `.csv` - Direct text reading

## Configuration

The system can be configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `DATA_DIR` | `data` | Directory for storing uploaded files and cache |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI language model for responses |
| `INDEX_DIR` | `data/index` | Directory for FAISS vector indices |

## Project Structure

```
rag-research-assistant/
├── src/
│   ├── cli/                 # Command-line interface
│   │   └── cli.py
│   ├── embeddings/          # OpenAI embedding client
│   │   └── client.py
│   ├── index/               # FAISS vector indexing
│   │   └── indexer.py
│   ├── ingest/              # Document parsing and chunking
│   │   ├── parser.py
│   │   └── chunker.py
│   ├── prompt/              # Prompt engineering
│   │   └── prompt.py
│   ├── retriever/           # Semantic search and retrieval
│   │   └── retriever.py
│   ├── ui/                  # Streamlit web interface
│   │   └── app.py
│   └── utils/               # Configuration and utilities
│       └── config.py
├── data/                    # Data storage
│   ├── embeddings_cache.json
│   └── index/
│       ├── faiss.index
│       └── metadata.pkl
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
└── README.md
```

## Examples

### Basic Workflow

1. **Ingest a research paper**:
```bash
python -m src.cli.cli ingest research_paper.pdf
```

2. **Ask questions about the paper**:
```bash
python -m src.cli.cli query "What methodology was used in this study?"
```

3. **Get detailed explanations**:
```bash
python -m src.cli.cli query "Explain the key findings and their implications" --top-k 7
```

### Web Interface Workflow

1. Start the web app: `streamlit run src/ui/app.py`
2. Upload your documents through the file uploader
3. Click "Ingest now" to process and index the documents
4. Enter your questions in the text input
5. View AI-generated answers with source citations

## Performance Notes

- **Embedding Caching**: Embeddings are cached to reduce API costs and improve performance
- **Chunk Size**: Text is chunked by tokens for optimal embedding quality
- **Vector Search**: FAISS provides fast similarity search even with large document collections
- **Memory Usage**: The system loads the entire vector index into memory for fast retrieval

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## Author

- [waltertaya](https://github.com/waltertaya)
