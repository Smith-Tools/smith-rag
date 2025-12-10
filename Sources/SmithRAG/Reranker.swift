import Foundation
import Logging
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Reranks search results using a cross-encoder model (via Ollama)
public actor Reranker {
    private let baseURL: String
    private let model: String
    private let logger = Logger(label: "smith-rag.reranker")
    
    public init(baseURL: String = "http://localhost:11434", model: String = "bge-reranker-base") {
        self.baseURL = baseURL
        self.model = model
    }
    
    /// Rerank candidates based on relevance to query
    /// Returns reordered results with new scores
    public func rerank(
        query: String,
        candidates: [(id: String, text: String, score: Float)],
        topK: Int
    ) async throws -> [(id: String, text: String, score: Float)] {
        // If reranker model not available, fall back to original scores
        guard await isAvailable() else {
            logger.warning("Reranker not available, using original scores")
            return Array(candidates.prefix(topK))
        }
        
        var reranked: [(id: String, text: String, score: Float)] = []
        
        for candidate in candidates {
            let score = try await scoreRelevance(query: query, document: candidate.text)
            reranked.append((candidate.id, candidate.text, score))
        }
        
        return reranked
            .sorted { $0.score > $1.score }
            .prefix(topK)
            .map { $0 }
    }
    
    /// Score relevance of a single document to a query
    private func scoreRelevance(query: String, document: String) async throws -> Float {
        let url = URL(string: "\(baseURL)/api/generate")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 10
        
        // Use the model's scoring capability
        // For models that don't support direct scoring, we prompt for relevance
        let prompt = """
        Query: \(query)
        Document: \(document.prefix(500))
        
        Rate relevance 0-10:
        """
        
        let body: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": ["num_predict": 5]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200,
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let responseText = json["response"] as? String else {
            return 0.5 // Neutral score on failure
        }
        
        // Parse score from response
        let score = parseScore(from: responseText)
        return score
    }
    
    private func parseScore(from text: String) -> Float {
        // Extract first number from response
        let pattern = #"(\d+(?:\.\d+)?)"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let range = Range(match.range(at: 1), in: text),
              let value = Float(text[range]) else {
            return 0.5
        }
        // Normalize to 0-1
        return min(max(value / 10.0, 0), 1)
    }
    
    /// Check if reranker model is available
    public func isAvailable() async -> Bool {
        guard let url = URL(string: "\(baseURL)/api/tags") else { return false }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let models = json["models"] as? [[String: Any]] else {
                return false
            }
            
            // Check if our model is in the list
            return models.contains { ($0["name"] as? String)?.contains(model) == true }
        } catch {
            return false
        }
    }
}
