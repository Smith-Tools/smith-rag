import Foundation
import Logging

/// Generates embeddings using Ollama via Python CLI (bypasses Swift URLSession issues)
public actor Embedder {
    private let pythonPath: String
    private let logger = Logger(label: "smith-rag.embedder")
    
    public init(baseURL: String = "http://localhost:11434", model: String = "nomic-embed-text") {
        // Use Python wrapper path (Script moved to repo)
        self.pythonPath = "/Volumes/Plutonian/_Developer/Smith-Tools/smith-rag/Scripts/ollama-embed.py"
    }
    
    /// Generate embedding vector for text
    public func embed(_ text: String, retries: Int = 3) async throws -> [Float] {
        var lastError: Error?
        
        for attempt in 0..<retries {
            do {
                return try embedOnce(text)
            } catch {
                lastError = error
                if attempt < retries - 1 {
                    try await Task.sleep(nanoseconds: UInt64(500_000_000 * (attempt + 1)))
                }
            }
        }
        
        throw lastError ?? EmbedderError.requestFailed
    }
    
    /// Single embed attempt via Python CLI
    private func embedOnce(_ text: String) throws -> [Float] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [pythonPath, text]
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        try process.run()
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorMsg = String(data: errorData, encoding: .utf8) ?? "unknown error"
            logger.error("Python embedding failed: \(errorMsg)")
            throw EmbedderError.requestFailed
        }
        
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        guard let floats = try? JSONDecoder().decode([Float].self, from: outputData) else {
            throw EmbedderError.invalidResponse
        }
        
        return floats
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
    
    /// Check if Python script exists
    public func isAvailable() async -> Bool {
        FileManager.default.fileExists(atPath: pythonPath)
    }
}

public enum EmbedderError: Error {
    case requestFailed
    case invalidResponse
    case ollamaNotAvailable
}
