import Foundation
import Logging
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Generates embeddings using Ollama or compatible API
public actor Embedder {
    private let baseURL: String
    private let model: String
    private let logger = Logger(label: "smith-rag.embedder")
    
    public init(baseURL: String = "http://localhost:11434", model: String = "nomic-embed-text") {
        self.baseURL = baseURL
        self.model = model
    }
    
    /// Generate embedding vector for text
    public func embed(_ text: String) async throws -> [Float] {
        let url = URL(string: "\(baseURL)/api/embeddings")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        
        let body: [String: Any] = [
            "model": model,
            "prompt": text
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw EmbedderError.requestFailed
        }
        
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double] else {
            throw EmbedderError.invalidResponse
        }
        
        return embedding.map { Float($0) }
    }
    
    /// Batch embed multiple texts
    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        var results: [[Float]] = []
        for text in texts {
            let vector = try await embed(text)
            results.append(vector)
        }
        return results
    }
    
    /// Check if Ollama is available
    public func isAvailable() async -> Bool {
        guard let url = URL(string: "\(baseURL)/api/tags") else { return false }
        
        do {
            let (_, response) = try await URLSession.shared.data(from: url)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }
}

public enum EmbedderError: Error {
    case requestFailed
    case invalidResponse
    case ollamaNotAvailable
}
