import Foundation
import Logging

/// Reranks search results using MLX embeddings for better relevance scoring
public actor MLXReranker {
    private let embedder: MLXEmbedder
    private let logger = Logger(label: "smith-rag.mlx-reranker")
    
    public init(embedder: MLXEmbedder) {
        self.embedder = embedder
    }
    
    /// Rerank candidates based on semantic similarity to query
    /// Uses embedding cosine similarity for scoring
    public func rerank(
        query: String,
        candidates: [(id: String, text: String, score: Float)],
        topK: Int
    ) async throws -> [(id: String, text: String, score: Float)] {
        guard !candidates.isEmpty else { return [] }
        
        let effectiveTopK = min(topK, candidates.count)
        
        logger.info("Reranking \(candidates.count) candidates for query")
        
        // Embed query
        let queryEmbedding = try await embedder.embed(query)
        guard !queryEmbedding.isEmpty else {
            logger.warning("Empty query embedding, returning original order")
            return Array(candidates.prefix(effectiveTopK))
        }
        
        // Embed all candidate texts
        let candidateTexts = candidates.map { $0.text }
        let candidateEmbeddings = try await embedder.embedBatch(candidateTexts)
        
        // Calculate cosine similarity for each candidate
        var scored: [(id: String, text: String, score: Float)] = []
        
        for (index, candidate) in candidates.enumerated() {
            guard index < candidateEmbeddings.count, !candidateEmbeddings[index].isEmpty else {
                scored.append((candidate.id, candidate.text, candidate.score))
                continue
            }
            
            let similarity = cosineSimilarity(queryEmbedding, candidateEmbeddings[index])
            scored.append((candidate.id, candidate.text, similarity))
        }
        
        // Sort by similarity score descending
        scored.sort { $0.score > $1.score }
        
        logger.info("Reranking complete, top score: \(scored.first?.score ?? 0)")
        
        return Array(scored.prefix(effectiveTopK))
    }
    
    /// Calculate cosine similarity between two vectors
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        let denominator = sqrt(normA) * sqrt(normB)
        guard denominator > 0 else { return 0 }
        
        return dotProduct / denominator
    }
    
    /// Check if reranker is available
    public func isAvailable() async -> Bool {
        await embedder.isAvailable()
    }
}
