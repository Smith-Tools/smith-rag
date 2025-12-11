import Foundation
import Logging

/// Main RAG engine that orchestrates search and fetch operations
public actor RAGEngine {
    private let store: ChunkStore
    private let embedder: Embedder
    private let vectorSearch: VectorSearch
    private let reranker: Reranker
    private let logger = Logger(label: "smith-rag.engine")
    
    /// Cached vectors for faster search
    private var vectorCache: [(id: String, vector: [Float])]?
    
    public init(
        databasePath: String,
        ollamaURL: String = "http://localhost:11434",
        embeddingModel: String = "nomic-embed-text",
        rerankerPath: String = "/Volumes/Plutonian/_models/jina-reranker/jina-rerank-cli.py"
    ) throws {
        self.store = try ChunkStore(databasePath: databasePath)
        self.embedder = Embedder(baseURL: ollamaURL, model: embeddingModel)
        self.vectorSearch = VectorSearch()
        self.reranker = Reranker(rerankerPath: rerankerPath)
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
        
        // 1. Embed the query
        let queryVector: [Float]
        do {
            queryVector = try await embedder.embed(query)
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
        
        // 3. Vector search
        let topCandidates = vectorSearch.search(
            query: queryVector,
            candidates: candidates,
            topK: candidateCount
        )
        
        // 4. Fetch text for top candidates
        var candidatesWithText: [(id: String, text: String, score: Float)] = []
        for (id, score) in topCandidates {
            if let chunk = try await store.fetchChunk(id: id) {
                candidatesWithText.append((id, chunk.text, score))
            }
        }
        
        // 5. Rerank (if enabled)
        let finalResults: [(id: String, text: String, score: Float)]
        if useReranker {
            finalResults = try await reranker.rerank(
                query: query,
                candidates: candidatesWithText,
                topK: limit
            )
        } else {
            finalResults = Array(candidatesWithText.prefix(limit))
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
    public func ingest(
        documentId: String,
        title: String,
        url: String?,
        content: String,
        chunkSize: Int = 500,
        overlap: Int = 50
    ) async throws {
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
                vector = try await embedder.embed(chunkText)
                // Small delay to avoid overwhelming Ollama
                try await Task.sleep(nanoseconds: 200_000_000) // 200ms
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
                let vector = try await embedder.embed(text)
                try await store.updateChunkVector(id: id, vector: vector)
                success += 1
                
                // Small delay to avoid overwhelming Ollama
                try await Task.sleep(nanoseconds: 200_000_000) // 200ms
            } catch {
                logger.warning("Failed to embed chunk \(id): \(error)")
            }
        }
        
        // Invalidate cache
        vectorCache = nil
        
        logger.info("Embedded \(success)/\(missingChunks.count) chunks")
        return success
    }
    
    /// Check if Ollama is available
    public func checkOllama() async -> (embedding: Bool, reranker: Bool) {
        async let embeddingAvailable = embedder.isAvailable()
        async let rerankerAvailable = reranker.isAvailable()
        return await (embeddingAvailable, rerankerAvailable)
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
