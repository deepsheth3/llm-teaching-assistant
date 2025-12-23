# 🎓 LLM Teaching Assistant v2

An AI-powered teaching assistant that retrieves research papers from arXiv, converts them into beginner-friendly lessons, and provides coding practice through LeetCode integration.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🔍 Semantic Paper Search** - Find relevant papers using natural language queries
- **📚 Lesson Generation** - Convert academic papers into beginner-friendly explanations
- **⚡ Streaming Support** - Real-time lesson generation via Server-Sent Events
- **💻 Coding Practice** - Random LeetCode problems for interview prep
- **🚀 Production Ready** - Proper error handling, logging, and caching

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                         │
│  /api/v1/teach  │  /api/v1/leetcode  │  /health                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      Teaching Service                           │
│            (Orchestrates all functionality)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Paper Service │   │Lesson Service │   │LeetCode Svc   │
│  - FAISS      │   │  - GPT-4o     │   │  - API        │
│  - GROBID     │   │  - Streaming  │   │  - Caching    │
│  - arXiv      │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/llm-teaching-assistant.git
cd llm-teaching-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

### 3. Initialize Index

```bash
python scripts/setup_index.py
```

### 4. Run Server

```bash
uvicorn api.main:app --reload
```

### 5. Test

```bash
# Health check
curl http://localhost:8000/health

# Generate a lesson
curl -X POST http://localhost:8000/api/v1/teach \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms in transformers"}'

# Get a coding problem
curl -X POST http://localhost:8000/api/v1/leetcode/random
```

## 📖 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/teach` | Generate a lesson about a topic |
| POST | `/api/v1/teach/stream` | Stream lesson generation (SSE) |
| POST | `/api/v1/search` | Search for papers |
| GET | `/api/v1/paper?url=...` | Get paper details |
| POST | `/api/v1/leetcode/random` | Get random coding problem |
| GET | `/api/v1/leetcode/problem/{slug}` | Get specific problem |
| GET | `/health` | Health check |

### Example Request

```python
import requests

# Generate a lesson
response = requests.post(
    "http://localhost:8000/api/v1/teach",
    json={
        "query": "how do transformers work",
        "difficulty": "beginner",
        "include_examples": True,
        "include_math": True
    }
)

lesson = response.json()
print(lesson["lesson"]["full_content"])
```

### Streaming Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/teach/stream",
    json={"query": "attention mechanisms"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

## 🛠️ Configuration

All configuration is done via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `API_PORT` | 8000 | Server port |
| `GROBID_URL` | cloud | GROBID service URL |
| `USE_GROBID` | true | Enable PDF parsing |
| `CACHE_ENABLED` | true | Enable caching |
| `LOG_LEVEL` | INFO | Logging level |

See `.env.example` for all options.

## 📁 Project Structure

```
llm-teaching-assistant-v2/
├── api/                    # FastAPI application
│   ├── main.py            # App entry point
│   └── routes/            # API endpoints
├── core/                   # Core utilities
│   ├── config.py          # Configuration
│   ├── exceptions.py      # Custom exceptions
│   └── logging.py         # Logging setup
├── models/                 # Pydantic models
│   ├── paper.py
│   ├── lesson.py
│   └── problem.py
├── services/               # Business logic
│   ├── teaching_service.py
│   ├── paper_service.py
│   ├── lesson_service.py
│   ├── leetcode_service.py
│   ├── embedding_service.py
│   └── cache_service.py
├── scripts/
│   └── setup_index.py     # Index initialization
├── data/                   # Data storage
│   ├── faiss/             # FAISS index
│   └── cache/             # File cache
├── .env.example
├── requirements.txt
└── README.md
```

## 🧪 Development

```bash
# Run with auto-reload
uvicorn api.main:app --reload

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy .
```

## 🐳 Docker (Optional)

```bash
# Build
docker build -t llm-teaching-assistant .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key llm-teaching-assistant
```

## 📊 Improvements Over v1

| Feature | v1 | v2 |
|---------|----|----|
| API | CLI only | REST API |
| Streaming | ❌ | ✅ SSE |
| Error Handling | Basic | Structured |
| Caching | ❌ | ✅ File + Memory |
| Logging | Print | Structured JSON |
| Config | Hardcoded | Environment vars |
| Types | Partial | Full Pydantic |
| GROBID Fallback | ❌ | ✅ Abstract mode |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) for the paper collection
- [GROBID](https://github.com/kermitt2/grobid) for PDF parsing
- [LeetCode](https://leetcode.com) for coding problems
