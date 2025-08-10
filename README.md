# AssistantHub

A flexible AI assistant framework built with langchain and Qdrant. It has multiple embedding options with multiple custom & efficient chunking, designed for internal knowledge-based chat applications.

> **Note:** This repository serves as the foundational framework for the business-focused [Virtual Assistant](https://sabbirhossainujjal.github.io/projects/w_virtual_assistant/) project, developed during my tenure at ACI Limited.

## Features

- **Multi-Provider Support**: OpenAI, Azure OpenAI, Google Gemini embeddings, Huggingface model embeedings
- **Vector Database**: Qdrant integration for efficient similarity search
- **Flexible Data Processing**: Support for PDF, DOCX, CSV, and text files
- **RESTful API**: FastAPI-powered endpoints with CORS support
- **Configurable**: YAML-based configuration system

## Project Structure

```
├── api.py                 # FastAPI application
├── config.yml             # Configuration file
├── utils/
│   ├── database_utils.py  # Vector database operations
│   ├── chat_utils.py      # Chat logic
│   └── data_utils.py      # Data processing utilities
├── Data/                  # Knowledge base files
├── scripts/               # Bash scripts
└── logs/                  # chat logs

```

## Quick Start

### Prerequisites

- Python 3.8+
- Qdrant server (local or cloud)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/sabbirhossainujjal/AssistantHub.git
cd AssistantHub
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Configure the application:

```bash
# Edit config.yml with your preferences
```

6. Start Qdrant (if running locally):

```bash
./scripts/database_start.sh
```

7. Run the API server:

```bash
python api.py
```

The API will be available at `http://localhost:8003`

## Configuration

Edit `config.yml` to customize:

- **Chat Model**: GPT-4, GPT-3.5, or other supported models
- **Embeddings**: Choose from OpenAI, Azure, or Gemini
- **Database**: Qdrant connection settings
- **System Prompt**: Customize AI behavior

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat_response` - Chat with the assistant

### Example Request

```json
{
  "query": "What is the price of noodles?",
  "language": "en"
}
```

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
QDRANT_API_KEY=your_qdrant_key  # if using Qdrant Cloud
AZURE_OPENAI_API_KEY=your_azure_key
GOOGLE_API_KEY=your_google_key
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.
