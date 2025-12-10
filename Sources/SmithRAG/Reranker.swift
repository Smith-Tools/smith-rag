import Foundation
import Logging
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Reranks search results using Jina Reranker v3 MLX (local, Apple Silicon optimized)
public actor Reranker {
    private let rerankerPath: String
    private let logger = Logger(label: "smith-rag.reranker")
    
    public init(rerankerPath: String = "/Volumes/Plutonian/_models/jina-reranker/jina-rerank-cli.py") {
        self.rerankerPath = rerankerPath
    }
    
    /// Rerank candidates based on relevance to query
    /// Returns reordered results with new scores
    public func rerank(
        query: String,
        candidates: [(id: String, text: String, score: Float)],
        topK: Int
    ) async throws -> [(id: String, text: String, score: Float)] {
        // Check if reranker is available
        guard FileManager.default.fileExists(atPath: rerankerPath) else {
            logger.warning("Jina reranker not found at \(rerankerPath), using original scores")
            return Array(candidates.prefix(topK))
        }
        
        // Prepare candidates JSON
        let candidateTexts = candidates.map { $0.text }
        guard let candidatesJSON = try? JSONSerialization.data(withJSONObject: candidateTexts),
              let candidatesString = String(data: candidatesJSON, encoding: .utf8) else {
            logger.warning("Failed to encode candidates, using original scores")
            return Array(candidates.prefix(topK))
        }
        
        // Call Python reranker
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            rerankerPath,
            "--query", query,
            "--candidates", candidatesString,
            "--top-k", String(topK)
        ]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8),
                  let resultData = output.data(using: .utf8),
                  let results = try? JSONDecoder().decode([RerankerResult].self, from: resultData) else {
                logger.warning("Failed to parse reranker output, using original scores")
                return Array(candidates.prefix(topK))
            }
            
            // Map results back to original IDs
            return results.map { result in
                let originalId = candidates[result.index].id
                return (originalId, result.text, result.score)
            }
            
        } catch {
            logger.warning("Reranker process failed: \(error), using original scores")
            return Array(candidates.prefix(topK))
        }
    }
    
    /// Check if Jina reranker is available
    public func isAvailable() async -> Bool {
        FileManager.default.fileExists(atPath: rerankerPath)
    }
}

private struct RerankerResult: Codable {
    let index: Int
    let text: String
    let score: Float
}

