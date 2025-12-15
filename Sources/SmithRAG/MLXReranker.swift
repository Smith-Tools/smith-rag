import Foundation
import Logging
import Accelerate

/// Reranks search results using stored embeddings for fast relevance scoring
/// Uses stored vectors from database - no model inference needed
public actor MLXReranker {
    private let embedder: MLXEmbedder
    private let logger = Logger(label: "smith-rag.mlx-reranker")
    
    public init(embedder: MLXEmbedder) {
        self.embedder = embedder
    }
    
    /// Rerank candidates using stored vectors (FAST - no MLX inference)
    /// queryVector: pre-computed query embedding
    /// candidates: (id, text, vector, score) - includes stored embedding
    public func rerankWithVectors(
        queryVector: [Float],
        candidates: [(id: String, text: String, vector: [Float], score: Float)],
        topK: Int
    ) async -> [(id: String, text: String, score: Float)] {
        guard !candidates.isEmpty, !queryVector.isEmpty else { return [] }
        
        let effectiveTopK = min(topK, candidates.count)
        
        logger.info("Reranking \(candidates.count) candidates with stored vectors (fast)")
        
        // Pre-compute query norm
        let queryNorm = vectorNorm(queryVector)
        guard queryNorm > 0 else {
            return Array(candidates.prefix(effectiveTopK).map { ($0.id, $0.text, $0.score) })
        }
        
        // Calculate cosine similarity using stored vectors - pure math, no inference
        var scored: [(id: String, text: String, score: Float)] = []
        scored.reserveCapacity(candidates.count)
        
        for candidate in candidates {
            guard !candidate.vector.isEmpty else {
                scored.append((candidate.id, candidate.text, candidate.score))
                continue
            }
            
            let similarity = cosineSimilarity(queryVector, candidate.vector, queryNorm: queryNorm)
            scored.append((candidate.id, candidate.text, similarity))
        }
        
        // Sort by similarity score descending
        scored.sort { $0.score > $1.score }
        
        logger.info("Reranking complete (fast), top score: \(scored.first?.score ?? 0)")
        
        return Array(scored.prefix(effectiveTopK))
    }
    
    /// Legacy rerank method that re-embeds candidates (SLOW - uses MLX inference)
    /// Kept for backward compatibility but not recommended
    public func rerank(
        query: String,
        candidates: [(id: String, text: String, score: Float)],
        topK: Int
    ) async throws -> [(id: String, text: String, score: Float)] {
        guard !candidates.isEmpty else { return [] }
        
        let effectiveTopK = min(topK, candidates.count)
        
        logger.info("Reranking \(candidates.count) candidates (slow - re-embedding)")
        
        // Embed query
        let queryEmbedding = try await embedder.embed(query)
        guard !queryEmbedding.isEmpty else {
            logger.warning("Empty query embedding, returning original order")
            return Array(candidates.prefix(effectiveTopK))
        }
        
        // Embed all candidate texts - THIS IS THE SLOW PART
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
        
        logger.info("Reranking complete (slow), top score: \(scored.first?.score ?? 0)")
        
        return Array(scored.prefix(effectiveTopK))
    }
    
    /// Calculate cosine similarity with pre-computed query norm (fast path)
    private func cosineSimilarity(_ a: [Float], _ b: [Float], queryNorm: Float) -> Float {
        guard a.count == b.count else { return 0 }
        
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        
        let bNorm = vectorNorm(b)
        guard bNorm > 0 else { return 0 }
        
        return dot / (queryNorm * bNorm)
    }
    
    /// Calculate cosine similarity (legacy, computes both norms)
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
    
    /// Compute vector L2 norm using vDSP
    private func vectorNorm(_ v: [Float]) -> Float {
        var sumOfSquares: Float = 0
        vDSP_svesq(v, 1, &sumOfSquares, vDSP_Length(v.count))
        return sqrt(sumOfSquares)
    }
    
    /// Check if reranker is available
    public func isAvailable() async -> Bool {
        await embedder.isAvailable()
    }
}
