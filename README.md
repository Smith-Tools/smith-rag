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

- **MLX-Native Embeddings**: Uses `Qwen3-Embedding-0.6B-4bit` (1024d) on Apple Silicon GPU
- **Instant Reranking**: Uses stored vectors for sub-second reranking (no model inference)
- **Offline-First**: No API calls, no cloud dependencies—all processing happens locally
- **12,500+ WWDC Chunks**: Covers sessions from 2014-2025
- **FTS5 Fallback**: Full-text search when vectors aren't available
- **WAL Mode**: Database optimized for concurrent read/write operations

## 📦 Installation

### Quick Install (copy-paste)

```bash
# 1. Clone and build
git clone https://github.com/Smith-Tools/smith-rag.git
cd smith-rag
xcodebuild -scheme smith-rag -configuration Release -destination 'platform=macOS' build

# 2. Install binary, Metal bundle, and wrapper script
mkdir -p ~/.smith/bin ~/.smith/rag
DERIVED=$(ls -d ~/Library/Developer/Xcode/DerivedData/smith-rag-*/Build/Products/Release 2>/dev/null | head -1)
cp "$DERIVED/smith-rag" ~/.smith/bin/
cp -R "$DERIVED/mlx-swift_Cmlx.bundle" ~/.smith/bin/

# Create wrapper (required for Metal library resolution)
cat > ~/.smith/bin/smith-rag-wrapper << 'EOF'
#!/bin/bash
cd ~/.smith/bin && exec ./smith-rag "$@"
EOF
chmod +x ~/.smith/bin/smith-rag-wrapper
sudo ln -sf ~/.smith/bin/smith-rag-wrapper /usr/local/bin/smith-rag

# 3. Download embedding model (~335MB)
pip install huggingface-hub
huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ

# 4. Verify
smith-rag --help
```

### What Gets Installed

| Path | Description |
|------|-------------|
| `~/.smith/bin/smith-rag` | Main binary |
| `~/.smith/bin/mlx-swift_Cmlx.bundle` | Metal shaders |
| `~/.smith/bin/smith-rag-wrapper` | Wrapper for correct CWD |
| `/usr/local/bin/smith-rag` | Symlink to wrapper |
| `~/.cache/huggingface/...` | MLX model files |

### Usage

```bash
smith-rag search "SwiftUI Observable" --database ~/.smith/rag/sosumi.db --limit 5
```

## 🤖 Agent Integration (Maxwell)

For Claude/Maxwell agent skills, add to your skill file:

```bash
# Step 1: ALWAYS search RAG first
smith-rag search "<query>" --database ~/.smith/rag/sosumi.db --limit 10
```

The agent can call this via Bash tool. Results are returned as scored chunks from WWDC transcripts.

### As Swift Package Dependency
```swift
.package(url: "https://github.com/Smith-Tools/smith-rag.git", from: "1.0.0")
```

## 🚀 Usage

### CLI Search
```bash
# Semantic search with MLX embeddings
smith-rag search "SwiftUI state management with @Observable" --database ~/.smith/rag/sosumi.db

# Exact-term search (FTS5)
smith-rag search "AnimationPlaybackController" --database ~/.smith/rag/sosumi.db --exact

# Semantic-only (skip keyword matches)
smith-rag search "SwiftUI animation" --database ~/.smith/rag/sosumi.db --semantic

# Limit results and skip reranking for speed
smith-rag search "Metal shader compilation" --limit 5 --no-rerank

# Use different model
smith-rag search "async/await patterns" --model nomic-ai/nomic-embed-text-v1.5
```

### Fetch & Status
```bash
smith-rag fetch <chunk-id> --database ~/.smith/rag/sosumi.db --mode context
smith-rag status --database ~/.smith/rag/sosumi.db
```

### Embedding Maintenance
Embedding refresh is handled by the source tools (e.g., `sosumi embed-missing`, `deadbeef embed-missing`).

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
