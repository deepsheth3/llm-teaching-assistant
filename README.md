# 🎓 LLM Teaching Assistant

<div align="center">

![Hero](https://img.shields.io/badge/AI-Powered_Learning-blue?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Transform dense research papers into lessons you'll actually understand.**

[Live Demo](https://your-app.railway.app) · [Report Bug](https://github.com/ganeshasrinivasd/llm-teaching-assistant/issues) · [Request Feature](https://github.com/ganeshasrinivasd/llm-teaching-assistant/issues)

</div>

---

## 🤔 The Problem

Ever tried reading a machine learning research paper?

```
"We propose a novel attention mechanism utilizing scaled dot-product 
attention with multi-head projections across the latent space..."
```

**Translation:** 😵‍💫

Research papers are written by experts, for experts. But what if you're:
- A student trying to learn ML
- A developer wanting to understand new techniques
- A curious mind exploring AI

You're stuck with two bad options:
1. **Read the paper** → Get lost in jargon, math, and assumptions
2. **Ask ChatGPT** → Get a generic summary that misses the nuances

---

## 💡 The Solution

What if an AI could:
1. **Find** the most relevant paper for what you want to learn
2. **Read** the entire paper (not just summarize the abstract)
3. **Teach** you section by section, like a patient tutor

That's exactly what this does.

```
You: "Teach me about attention mechanisms"

AI: *finds the Transformer paper*
    *reads all 15 pages*
    *generates a personalized lesson*
    
    "Let's start with WHY attention matters. Imagine you're 
    translating 'The cat sat on the mat' to French. When 
    translating 'cat', which English words should you focus on?
    
    This is attention - letting the model CHOOSE what to look at..."
```

---

## 🧠 Why Not Just Use ChatGPT?

Great question. Here's the difference:

### ChatGPT Approach
```
You: "Explain transformers"
ChatGPT: *searches its training data*
         *gives you a general explanation*
         *might be outdated or incomplete*
```

### Our Approach
```
You: "Explain transformers"
Us:  1. Search 231 curated ML papers using semantic similarity
     2. Find the ACTUAL paper that best matches your query
     3. Download the PDF
     4. Parse it into structured sections using GROBID
     5. Generate lessons from the REAL content
     6. Cite the source so you can verify
```

### Technical Comparison

| Aspect | ChatGPT | LLM Teaching Assistant |
|--------|---------|------------------------|
| **Source** | Training data (static) | Live papers (dynamic) |
| **Accuracy** | May hallucinate | Grounded in real papers |
| **Depth** | Surface-level | Section-by-section deep dive |
| **Citation** | None | Links to original paper |
| **Recency** | Knowledge cutoff | Always current papers |
| **Customization** | Generic | Adapts to your level |

### Non-Technical Explanation

Think of it like this:

**ChatGPT** = A friend who read a lot of books and tells you what they remember

**Us** = A librarian who:
- Finds the exact book you need
- Reads it cover to cover
- Explains each chapter in simple terms
- Shows you where to find the original

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│                         (React + TypeScript)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    Hero     │  │   Lesson    │  │   Problem   │  │   Theme     │     │
│  │   Input     │  │   Display   │  │   Display   │  │   Toggle    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ HTTP/REST
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND                                     │
│                           (FastAPI + Python)                             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      API Layer (/api/v1)                          │   │
│  │  ┌─────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ /health │  │   /teach    │  │ /teach/stream│  │ /leetcode  │  │   │
│  │  └─────────┘  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Service Layer                                 │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐     │   │
│  │  │   Teaching   │  │    Paper      │  │      Lesson        │     │   │
│  │  │   Service    │──│   Service     │──│      Service       │     │   │
│  │  │ (orchestrate)│  │ (fetch+parse) │  │ (generate lessons) │     │   │
│  │  └──────────────┘  └───────────────┘  └────────────────────┘     │   │
│  │         │                  │                     │                │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐     │   │
│  │  │   LeetCode   │  │   Embedding   │  │      Cache         │     │   │
│  │  │   Service    │  │   Service     │  │      Service       │     │   │
│  │  └──────────────┘  └───────────────┘  └────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   FAISS       │          │     GROBID      │          │    OpenAI       │
│   Vector DB   │          │  (PDF Parser)   │          │     API         │
│               │          │                 │          │                 │
│ 231 papers    │          │ Extracts        │          │ • Embeddings    │
│ indexed by    │          │ sections from   │          │ • GPT-4o-mini   │
│ semantic      │          │ academic PDFs   │          │   for lessons   │
│ similarity    │          │                 │          │                 │
└───────────────┘          └─────────────────┘          └─────────────────┘
        ▲                            ▲
        │                            │
┌───────────────┐          ┌─────────────────┐
│    arXiv      │          │    LeetCode     │
│    Papers     │          │      API        │
│               │          │                 │
│ Source of     │          │ Coding problems │
│ ML research   │          │ for practice    │
└───────────────┘          └─────────────────┘
```

---

## 🔄 How It Works (Flow)

```
                                    ┌─────────────────┐
                                    │   User Query    │
                                    │ "Explain BERT"  │
                                    └────────┬────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   1. EMBED THE QUERY     │
                              │   OpenAI text-embedding  │
                              │   → 1536-dim vector      │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   2. SEMANTIC SEARCH     │
                              │   FAISS finds closest    │
                              │   paper from 231 indexed │
                              │   → arxiv.org/abs/xxx    │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   3. FETCH & PARSE PDF   │
                              │   Download from arXiv    │
                              │   GROBID extracts:       │
                              │   • Introduction         │
                              │   • Methods              │
                              │   • Results              │
                              │   • 20+ sections         │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   4. GENERATE LESSONS    │
                              │   For each section:      │
                              │   GPT-4o-mini creates    │
                              │   beginner-friendly      │
                              │   explanation            │
                              └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │   5. RETURN LESSON       │
                              │   Complete course with:  │
                              │   • Table of contents    │
                              │   • Section-by-section   │
                              │   • Source citation      │
                              │   • Estimated read time  │
                              └──────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose | Why This? |
|------------|---------|-----------|
| **FastAPI** | REST API | Async, fast, auto-docs, Python type hints |
| **FAISS** | Vector search | Facebook's library, blazing fast similarity search |
| **GROBID** | PDF parsing | Best-in-class academic PDF parser, extracts structure |
| **OpenAI** | Embeddings + LLM | text-embedding-3-small + GPT-4o-mini |
| **Pydantic** | Data validation | Type safety, automatic serialization |

### Frontend
| Technology | Purpose | Why This? |
|------------|---------|-----------|
| **React 18** | UI framework | Component-based, huge ecosystem |
| **TypeScript** | Type safety | Catch errors at compile time |
| **Tailwind CSS** | Styling | Utility-first, rapid development |
| **Framer Motion** | Animations | Smooth, declarative animations |
| **Vite** | Build tool | Lightning fast HMR |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| **Railway** | Hosting (backend + frontend) |
| **GROBID Cloud** | PDF parsing service |
| **GitHub** | Version control |

---

## 📁 Project Structure

```
llm-teaching-assistant/
│
├── backend/                          # Python FastAPI backend
│   ├── api/
│   │   ├── main.py                   # FastAPI app entry
│   │   └── routes/
│   │       ├── teach.py              # /teach endpoints
│   │       ├── leetcode.py           # /leetcode endpoints
│   │       └── health.py             # Health checks
│   │
│   ├── services/
│   │   ├── teaching_service.py       # Main orchestration
│   │   ├── paper_service.py          # Paper fetching + GROBID
│   │   ├── lesson_service.py         # GPT lesson generation
│   │   ├── embedding_service.py      # FAISS + OpenAI embeddings
│   │   ├── leetcode_service.py       # LeetCode integration
│   │   └── cache_service.py          # Caching layer
│   │
│   ├── models/                       # Pydantic data models
│   ├── core/                         # Config, logging, exceptions
│   └── requirements.txt
│
├── frontend/                         # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Hero.tsx              # Main input section
│   │   │   ├── LessonDisplay.tsx     # Lesson modal
│   │   │   ├── ProblemDisplay.tsx    # LeetCode modal
│   │   │   └── Header.tsx            # Navigation
│   │   ├── lib/
│   │   │   └── api.ts                # API client
│   │   └── App.tsx                   # Main app
│   │
│   └── package.json
│
└── README.md                         # You are here
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key

### 1. Clone & Setup Backend

```bash
git clone https://github.com/ganeshasrinivasd/llm-teaching-assistant.git
cd llm-teaching-assistant/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Initialize the paper index
python scripts/setup_index.py

# Run the server
uvicorn api.main:app --reload
```

### 2. Setup Frontend

```bash
cd ../frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### 3. Open App

Visit **http://localhost:3000** 🎉

---

## 📖 API Reference

### Generate Lesson
```http
POST /api/v1/teach
Content-Type: application/json

{
  "query": "Explain attention mechanisms",
  "difficulty": "beginner",
  "max_sections": 5
}
```

### Get Coding Problem
```http
POST /api/v1/leetcode/random
Content-Type: application/json

{
  "difficulties": ["Medium", "Hard"]
}
```

### Health Check
```http
GET /health
```

Full API docs available at `/docs` when running locally.

---

## 🎯 Features

- [x] Semantic paper search
- [x] PDF parsing with GROBID
- [x] Section-by-section lessons
- [x] LeetCode integration
- [x] Dark/Light mode
- [x] Mobile responsive
- [ ] Streaming responses (coming soon)
- [ ] User accounts
- [ ] Save lesson history
- [ ] Multiple difficulty levels

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) for the curated paper collection
- [GROBID](https://github.com/kermitt2/grobid) for PDF parsing
- [OpenAI](https://openai.com) for embeddings and language models
- [LeetCode](https://leetcode.com) for coding problems

---

<div align="center">

**Built with ❤️ for learners everywhere**

[⬆ Back to top](#-llm-teaching-assistant)

</div>
