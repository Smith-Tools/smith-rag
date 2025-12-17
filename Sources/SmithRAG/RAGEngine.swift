import Foundation
import Logging

/// Backend configuration for RAG engine
public enum RAGBackend: Sendable {
    case ollama(url: String, model: String)
    case mlx(modelId: String)
}

/// Main RAG engine that orchestrates search and fetch operations
public actor RAGEngine {
    private let store: ChunkStore
    private let vectorSearch: VectorSearch
    private let logger = Logger(label: "smith-rag.engine")
    
    // Backend-specific components
    private let backend: RAGBackend
    private var ollamaEmbedder: Embedder?
    private var ollamaReranker: Reranker?
    private var mlxEmbedder: MLXEmbedder?
    private var mlxReranker: MLXReranker?
    
    /// Cached vectors for faster search
    private var vectorCache: [(id: String, vector: [Float])]?
    
    /// Initialize with Ollama backend (legacy)
    public init(
        databasePath: String,
        ollamaURL: String = "http://localhost:11434",
        embeddingModel: String = "nomic-embed-text",
        rerankerPath: String = "/Volumes/Plutonian/_models/jina-reranker/jina-rerank-cli.py"
    ) throws {
        self.store = try ChunkStore(databasePath: databasePath)
        self.vectorSearch = VectorSearch()
        self.backend = .ollama(url: ollamaURL, model: embeddingModel)
        self.ollamaEmbedder = Embedder(baseURL: ollamaURL, model: embeddingModel)
        self.ollamaReranker = Reranker(rerankerPath: rerankerPath)
    }
    
    /// Initialize with MLX backend (recommended for Apple Silicon)
    public init(
        databasePath: String,
        mlxModelId: String = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    ) throws {
        self.store = try ChunkStore(databasePath: databasePath)
        self.vectorSearch = VectorSearch()
        self.backend = .mlx(modelId: mlxModelId)
        self.mlxEmbedder = MLXEmbedder(modelId: mlxModelId)
        // MLXReranker will be initialized lazily when needed
    }
    
    /// Get the current embedder (lazy MLX reranker init)
    private func ensureMLXReranker() async {
        if mlxReranker == nil, let embedder = mlxEmbedder {
            mlxReranker = MLXReranker(embedder: embedder)
        }
    }

    
    // MARK: - Search
    
    /// Search for relevant chunks
    /// Returns: Array of (id, title, snippet, score)
    public func search(
        query: String,
        limit: Int = 5,
        candidateCount: Int = 100,
        useReranker: Bool = true
    ) async throws -> [SearchResult] {
        logger.info("Searching for: \(query)")
        
        // 1. Embed the query (backend-agnostic)
        let queryVector: [Float]
        do {
            queryVector = try await embedQuery(query)
        } catch {
            logger.warning("Embedding failed, falling back to FTS5: \(error)")
            return try await fallbackSearch(query: query, limit: limit)
        }
        
        // 2. Load vectors (cached)
        if vectorCache == nil {
            vectorCache = try await store.fetchAllVectors()
        }
        
        guard let candidates = vectorCache, !candidates.isEmpty else {
            logger.warning("No vectors in database, falling back to FTS5")
            return try await fallbackSearch(query: query, limit: limit)
        }
        
        // 3. Vector search (get vectors for reranking)
        let topCandidates = vectorSearch.searchWithVectors(
            query: queryVector,
            candidates: candidates,
            topK: candidateCount
        )
        
        // 4. Fetch text for top candidates (keeping vectors for rerank)
        var candidatesWithTextAndVectors: [(id: String, text: String, vector: [Float], score: Float)] = []
        for (id, vector, score) in topCandidates {
            if let chunk = try await store.fetchChunk(id: id) {
                candidatesWithTextAndVectors.append((id, chunk.text, vector, score))
            }
        }
        
        // 5. Rerank using stored vectors (FAST - no MLX inference)
        let finalResults: [(id: String, text: String, score: Float)]
        if useReranker {
            finalResults = try await rerankWithStoredVectors(
                queryVector: queryVector,
                candidates: candidatesWithTextAndVectors,
                topK: limit
            )
        } else {
            finalResults = Array(candidatesWithTextAndVectors.prefix(limit).map { ($0.id, $0.text, $0.score) })
        }
        
        // 6. Format results
        return finalResults.map { result in
            SearchResult(
                id: result.id,
                snippet: String(result.text.prefix(300)),
                score: result.score
            )
        }
    }
    
    /// Fallback search using FTS5 (keyword-based)
    private func fallbackSearch(query: String, limit: Int) async throws -> [SearchResult] {
        let results = try await store.keywordSearch(query: query, limit: limit)
        return results.map { SearchResult(id: $0.id, snippet: $0.snippet, score: 0.5) }
    }
    
    // MARK: - Backend Helpers
    
    /// Embed query using the configured backend
    private func embedQuery(_ text: String) async throws -> [Float] {
        switch backend {
        case .ollama:
            guard let embedder = ollamaEmbedder else {
                throw RAGError.embeddingFailed
            }
            return try await embedder.embed(text)
        case .mlx:
            guard let embedder = mlxEmbedder else {
                throw RAGError.embeddingFailed
            }
            return try await embedder.embed(text)
        }
    }
    
    /// Embed text for storage using the configured backend
    private func embedText(_ text: String) async throws -> [Float] {
        return try await embedQuery(text)
    }
    
    /// Rerank candidates using the configured backend
    private func rerankCandidates(
        query: String,
        candidates: [(id: String, text: String, score: Float)],
        topK: Int
    ) async throws -> [(id: String, text: String, score: Float)] {
        switch backend {
        case .ollama:
            guard let reranker = ollamaReranker else {
                return Array(candidates.prefix(topK))
            }
            return try await reranker.rerank(query: query, candidates: candidates, topK: topK)
        case .mlx:
            await ensureMLXReranker()
            guard let reranker = mlxReranker else {
                return Array(candidates.prefix(topK))
            }
            return try await reranker.rerank(query: query, candidates: candidates, topK: topK)
        }
    }
    
    /// Fast rerank using stored vectors (no MLX inference needed)
    private func rerankWithStoredVectors(
        queryVector: [Float],
        candidates: [(id: String, text: String, vector: [Float], score: Float)],
        topK: Int
    ) async throws -> [(id: String, text: String, score: Float)] {
        switch backend {
        case .ollama:
            // Ollama doesn't support vector-based reranking, fall back to legacy
            guard let reranker = ollamaReranker else {
                return Array(candidates.prefix(topK).map { ($0.id, $0.text, $0.score) })
            }
            let legacyCandidates = candidates.map { ($0.id, $0.text, $0.score) }
            return try await reranker.rerank(query: "", candidates: legacyCandidates, topK: topK)
        case .mlx:
            await ensureMLXReranker()
            guard let reranker = mlxReranker else {
                return Array(candidates.prefix(topK).map { ($0.id, $0.text, $0.score) })
            }
            // Use new fast rerank with stored vectors
            return await reranker.rerankWithVectors(
                queryVector: queryVector,
                candidates: candidates,
                topK: topK
            )
        }
    }
    
    // MARK: - Fetch
    
    /// Fetch content by chunk ID
    public func fetch(
        id: String,
        mode: FetchMode = .chunk,
        contextSize: Int = 2
    ) async throws -> FetchResult {
        switch mode {
        case .chunk:
            guard let chunk = try await store.fetchChunk(id: id) else {
                throw RAGError.notFound(id)
            }
            return FetchResult(id: id, content: chunk.text, mode: .chunk)
            
        case .context:
            let chunks = try await store.fetchChunksWithContext(chunkId: id, contextSize: contextSize)
            return FetchResult(id: id, content: chunks.joined(separator: "\n\n"), mode: .context)
            
        case .full:
            guard let chunk = try await store.fetchChunk(id: id) else {
                throw RAGError.notFound(id)
            }
            guard let doc = try await store.fetchDocument(id: chunk.documentId) else {
                throw RAGError.notFound(chunk.documentId)
            }
            return FetchResult(id: id, content: doc.content, mode: .full)
        }
    }
    
    // MARK: - Ingestion
    
    /// Ingest a document (chunking + embedding)
    /// - Returns: `true` if document was ingested, `false` if skipped (already exists)
    @discardableResult
    public func ingest(
        documentId: String,
        title: String,
        url: String?,
        content: String,
        chunkSize: Int = 500,
        overlap: Int = 50,
        skipIfExists: Bool = false
    ) async throws -> Bool {
        // Skip if document already exists
        if skipIfExists {
            if let _ = try await store.fetchDocument(id: documentId) {
                logger.info("Skipping existing document: \(title)")
                return false
            }
        }
        
        logger.info("Ingesting document: \(title)")
        
        // 1. Store document
        try await store.insertDocument(id: documentId, title: title, url: url, content: content)
        
        // 2. Chunk content
        let chunks = chunkText(content, size: chunkSize, overlap: overlap)
        
        // 3. Embed and store chunks
        for (index, chunkText) in chunks.enumerated() {
            let chunkId = "\(documentId)-chunk-\(index)"
            
            var vector: [Float]?
            do {
                vector = try await embedText(chunkText)
                // Small delay for Ollama only (MLX doesn't need it)
                if case .ollama = backend {
                    try await Task.sleep(nanoseconds: 2_000_000_000) // 2s
                }
            } catch {
                logger.warning("Failed to embed chunk \(index): \(error)")
            }
            
            try await store.insertChunk(
                id: chunkId,
                documentId: documentId,
                index: index,
                text: chunkText,
                vector: vector
            )
        }
        
        // 4. Invalidate vector cache
        vectorCache = nil
        
        logger.info("Ingested \(chunks.count) chunks for \(title)")
        return true
    }
    
    /// Split text into overlapping chunks
    private func chunkText(_ text: String, size: Int, overlap: Int) -> [String] {
        let words = text.split(separator: " ")
        var chunks: [String] = []
        var start = 0
        
        while start < words.count {
            let end = min(start + size, words.count)
            let chunk = words[start..<end].joined(separator: " ")
            chunks.append(chunk)
            start += size - overlap
        }
        
        return chunks
    }
    
    // MARK: - Status
    
    /// Re-embed chunks that are missing vectors
    public func embedMissing(batchSize: Int = 100) async throws -> Int {
        logger.info("Finding chunks without embeddings...")
        
        let missingChunks = try await store.fetchChunksWithoutVectors(limit: batchSize)
        logger.info("Found \(missingChunks.count) chunks without vectors")
        
        var success = 0
        for (id, text) in missingChunks {
            do {
                let vector = try await embedText(text)
                try await store.updateChunkVector(id: id, vector: vector)
                success += 1
                
                // Small delay for Ollama only (MLX doesn't need it)
                if case .ollama = backend {
                    try await Task.sleep(nanoseconds: 2_000_000_000) // 2s
                }
            } catch {
                logger.warning("Failed to embed chunk \(id): \(error)")
            }
        }
        
        // Invalidate cache
        vectorCache = nil
        
        logger.info("Embedded \(success)/\(missingChunks.count) chunks")
        return success
    }
    
    /// Check if backend is available
    public func checkBackend() async -> (embedding: Bool, reranker: Bool) {
        switch backend {
        case .ollama:
            async let embeddingAvailable = ollamaEmbedder?.isAvailable() ?? false
            async let rerankerAvailable = ollamaReranker?.isAvailable() ?? false
            return await (embeddingAvailable, rerankerAvailable)
        case .mlx:
            async let embeddingAvailable = mlxEmbedder?.isAvailable() ?? false
            await ensureMLXReranker()
            async let rerankerAvailable = mlxReranker?.isAvailable() ?? false
            return await (embeddingAvailable, rerankerAvailable)
        }
    }
    
    /// Legacy method for backward compatibility
    public func checkOllama() async -> (embedding: Bool, reranker: Bool) {
        return await checkBackend()
    }
    
    /// Re-embed all chunks with current backend (for migration)
    /// Returns: (success count, failure count, total)
    public func reembedAll(
        batchSize: Int = 50,
        progressCallback: ((Int, Int) async -> Void)? = nil
    ) async throws -> (success: Int, failed: Int, total: Int) {
        logger.info("Starting full re-embedding with MLX...")
        
        // Fetch all chunks
        let allChunks = try await store.fetchAllChunksForReembedding()
        let total = allChunks.count
        logger.info("Found \(total) chunks to re-embed")
        
        var success = 0
        var failed = 0
        
        // Process in batches
        for (index, (id, text)) in allChunks.enumerated() {
            do {
                let vector = try await embedText(text)
                try await store.updateChunkVector(id: id, vector: vector)
                success += 1
            } catch {
                logger.warning("Failed to embed chunk \(id): \(error)")
                failed += 1
            }
            
            // Report progress every batch
            if (index + 1) % batchSize == 0 || index == allChunks.count - 1 {
                await progressCallback?(index + 1, total)
            }
        }
        
        // Invalidate cache
        vectorCache = nil
        
        logger.info("Re-embedding complete: \(success) success, \(failed) failed out of \(total)")
        return (success, failed, total)
    }
}

// MARK: - Types

public struct SearchResult: Codable, Sendable {
    public let id: String
    public let snippet: String
    public let score: Float
}

public struct FetchResult: Codable, Sendable {
    public let id: String
    public let content: String
    public let mode: FetchMode
}

public enum FetchMode: String, Codable, Sendable {
    case chunk
    case context
    case full
}

public enum RAGError: Error {
    case notFound(String)
    case embeddingFailed
    case searchFailed
}
