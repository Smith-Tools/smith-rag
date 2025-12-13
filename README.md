# SmithRAG

> **Semantic Search & RAG Engine for Apple Developer Knowledge**

SmithRAG is the **retrieval-augmented generation (RAG) engine** that powers semantic search across Apple developer documentation, WWDC transcripts, and third-party Swift resources. It enables both agents and developers to find contextually relevant information using natural language queries.

## 🎯 Role in Smith Tools Ecosystem

```
Developer/Agent asks: "How do I implement @Observable in SwiftUI?"
        ↓
┌────────────────────────────────────────────────────────────────────┐
│                        SmithRAG                                     │
├─────────────────────────────────────────────────────────────────────┤
│  📚 sosumi database     → Apple docs, WWDC sessions (2014-2024)    │
│  📖 maxwell database    → Your personal learnings & discoveries    │
│  📦 scully database     → Third-party package documentation        │
│                                                                     │
│  🔍 Semantic Search     → Understands meaning, not just keywords   │
│  🏆 MLX Reranking       → Surfaces most relevant results first     │
│  ⚡ 1024d Embeddings    → High-quality vector representations      │
└─────────────────────────────────────────────────────────────────────┘
        ↓
Top 10 contextually relevant chunks from official Apple sources
```

SmithRAG bridges the gap between **raw documentation** and **actionable context**. Instead of searching through hundreds of pages, you get precisely the information needed for your implementation task.

## ✨ Key Features

- **MLX-Native Embeddings**: Uses `Qwen3-Embedding-0.6B-4bit` (1024d) running entirely on Apple Silicon GPU
- **Offline-First**: No API calls, no cloud dependencies—all processing happens locally
- **Embedding-Based Reranking**: Results are sorted by semantic relevance, not just keyword matching
- **FTS5 Fallback**: Full-text search when vectors aren't available
- **Multiple Backends**: Supports MLX (recommended) and Ollama
- **WAL Mode**: Database optimized for concurrent read/write operations

## 📦 Installation

### From Source
```bash
git clone https://github.com/Smith-Tools/smith-rag.git
cd smith-rag

# Build with xcodebuild (required for Metal shaders)
xcodebuild -scheme rag -destination 'platform=macOS' build

# Download the embedding model (one-time, ~600MB)
huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
```

### As Swift Package Dependency
```swift
.package(url: "https://github.com/Smith-Tools/smith-rag.git", from: "1.0.0")
```

## 🚀 Usage

### CLI Search
```bash
# Semantic search with MLX embeddings
rag search "SwiftUI state management with @Observable" --database ~/.smith/rag/sosumi.db

# Limit results and skip reranking for speed
rag search "Metal shader compilation" --limit 5 --no-rerank

# Use different model
rag search "async/await patterns" --model nomic-ai/nomic-embed-text-v1.5
```

### Migration (Re-embedding)
```bash
# Re-embed all chunks with Qwen3 (1024d vectors)
rag migrate --confirm --database ~/.smith/rag/sosumi.db
```

### Swift API
```swift
import SmithRAG

// Initialize with MLX backend
let engine = try RAGEngine(databasePath: "~/.smith/rag/sosumi.db")

// Semantic search
let results = try await engine.search(
    query: "How to use @Observable macro",
    limit: 10,
    candidateMultiplier: 3
)

for result in results {
    print("[\(result.score)] \(result.title)")
    print(result.content)
}
```

## 🗄️ Database Schema

SmithRAG stores chunks with their vector embeddings:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | Unique chunk identifier |
| `doc_id` | TEXT | Parent document ID |
| `title` | TEXT | Chunk title/heading |
| `content` | TEXT | Full text content |
| `vector` | BLOB | 1024d float32 embedding |
| `metadata` | JSON | Source URL, year, type |

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAGEngine                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ MLXEmbedder │  │ VectorSearch │  │     MLXReranker        │  │
│  │ (Qwen3)     │  │ (Cosine Sim) │  │ (Embedding Similarity) │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│         │                │                      │                │
│         └────────────────┴──────────────────────┘                │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │                    ChunkStore (GRDB)                       │  │
│  │         SQLite with FTS5 + Vector Storage                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Load Time | ~3-4 seconds (cached) |
| Embedding Speed | ~1.5 chunks/second |
| Vector Dimension | 1024 (Qwen3) |
| Context Window | 2048 tokens (capped for speed) |
| Batch Size | 64 chunks |

## 🤝 Integration with Other Tools

- **sosumi**: Ingests Apple documentation and WWDC transcripts → SmithRAG indexes them
- **maxwell**: Stores personal learnings → SmithRAG makes them searchable
- **scully**: Extracts package docs → SmithRAG enables semantic lookup
- **smith-cli**: Orchestrates searches across all knowledge bases

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Part of the [Smith Tools](https://github.com/Smith-Tools) ecosystem**
*Contextual intelligence for Swift development*
