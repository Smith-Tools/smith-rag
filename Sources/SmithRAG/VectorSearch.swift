import Foundation
import Accelerate

/// Fast vector similarity search using vDSP/Accelerate
public struct VectorSearch {
    
    public init() {}
    
    /// Find top-K most similar vectors using cosine similarity
    public func search(
        query: [Float],
        candidates: [(id: String, vector: [Float])],
        topK: Int
    ) -> [(id: String, score: Float)] {
        var results: [(id: String, score: Float)] = []
        results.reserveCapacity(candidates.count)
        
        // Pre-compute query norm
        let queryNorm = vectorNorm(query)
        guard queryNorm > 0 else { return [] }
        
        for (id, vector) in candidates {
            let score = cosineSimilarity(query, vector, queryNorm: queryNorm)
            results.append((id, score))
        }
        
        // Sort by score descending, take top-K
        return results
            .sorted { $0.score > $1.score }
            .prefix(topK)
            .map { $0 }
    }
    
    /// Compute cosine similarity with pre-computed query norm
    private func cosineSimilarity(_ a: [Float], _ b: [Float], queryNorm: Float) -> Float {
        guard a.count == b.count else { return 0 }
        
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        
        let bNorm = vectorNorm(b)
        guard bNorm > 0 else { return 0 }
        
        return dot / (queryNorm * bNorm)
    }
    
    /// Compute vector L2 norm using vDSP
    private func vectorNorm(_ v: [Float]) -> Float {
        var sumOfSquares: Float = 0
        vDSP_svesq(v, 1, &sumOfSquares, vDSP_Length(v.count))
        return sqrt(sumOfSquares)
    }
    
    /// Normalize a vector to unit length
    public func normalize(_ vector: [Float]) -> [Float] {
        var length: Float = 0
        vDSP_svesq(vector, 1, &length, vDSP_Length(vector.count))
        length = sqrt(length)
        
        guard length > 0 else { return vector }
        
        var result = [Float](repeating: 0, count: vector.count)
        var divisor = length
        vDSP_vsdiv(vector, 1, &divisor, &result, 1, vDSP_Length(vector.count))
        return result
    }
}
