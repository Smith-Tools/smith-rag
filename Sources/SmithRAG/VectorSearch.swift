import Foundation
import Accelerate

/// Fast vector similarity search using vDSP/Accelerate
public struct VectorSearch {
    
    public init() {}
    
    /// Find top-K most similar vectors using cosine similarity
    /// Assumes vectors are normalized (nomic-embed-text outputs normalized vectors)
    public func search(
        query: [Float],
        candidates: [(id: String, vector: [Float])],
        topK: Int
    ) -> [(id: String, score: Float)] {
        var results: [(id: String, score: Float)] = []
        results.reserveCapacity(candidates.count)
        
        for (id, vector) in candidates {
            // Cosine similarity for normalized vectors = dot product
            let score = dotProduct(query, vector)
            results.append((id, score))
        }
        
        // Sort by score descending, take top-K
        return results
            .sorted { $0.score > $1.score }
            .prefix(topK)
            .map { $0 }
    }
    
    /// Compute dot product using vDSP (Apple Accelerate framework)
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
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
