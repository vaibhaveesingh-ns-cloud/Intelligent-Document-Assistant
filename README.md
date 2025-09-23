# Intelligent Document Assistant

A React + FastAPI application for document Q&A with vector search, conversation memory, and source citations.

## Features

- **Multi-format Document Upload**: PDF, DOCX, PPTX, TXT, MD, CSV/XLSX, PNG/JPG
- **Vector Database**: ChromaDB with persistent storage
- **LangChain Integration**: Document processing and Q&A
- **Conversation Memory**: Maintains context across questions
- **Source Citations**: Shows which documents were used for answers
- **OpenAI Integration**: Optional LLM-powered responses
- **Docker Support**: Easy deployment with docker-compose

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key for LLM features

### Environment Setup

1. **Configure API Key**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=sk-your-actual-key-here
   ```

### Running with Docker

1. Clone the repository and navigate to the project directory
2. Set up your `.env` file (see Environment Setup above)
3. Start the services:
   ```bash
   docker compose up --build
   ```
4. Open your browser to:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

### Running Locally (Development)

#### Backend
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (see Environment Setup above)

# Run the Streamlit app
streamlit run app.py

# OR run the FastAPI server
python run_backend.py
# OR
uvicorn backend.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/config` - Configuration and supported file types
- `POST /api/documents` - Upload and index documents
- `POST /api/chat` - Ask questions about uploaded documents

## Configuration

The application uses environment variables for configuration. Create a `.env` file in the project root:

```bash
# Required: OpenAI API key for LLM features
OPENAI_API_KEY=sk-your-actual-key-here

# Optional: OpenAI model (defaults to gpt-4o-mini)
OPENAI_MODEL=gpt-4o-mini

# Frontend configuration
VITE_API_BASE_URL=  # Frontend API base URL (defaults to empty for proxy)
```

**Security Note**: The `.env` file is automatically excluded from version control via `.gitignore` to protect your API keys.

## Docker Volumes

- `chroma_data` - Persistent storage for the vector database

## Architecture

- **Backend**: FastAPI with LangChain, ChromaDB, and document processors
- **Frontend**: React with Vite, modern UI components
- **Vector Store**: ChromaDB for semantic search
- **Document Processing**: Support for multiple file formats with OCR
- **CORS**: Properly configured for local development

## Development

The frontend uses Vite with a proxy configuration to avoid CORS issues during development. The backend includes comprehensive error handling and supports both local and OpenAI embeddings.

## Deployment

For production deployment, consider:
- Using a reverse proxy (nginx)
- Setting up proper SSL certificates
- Configuring environment variables securely
- Using a managed database service
- Setting up monitoring and logging
