import ArgumentParser
import Foundation
import SmithRAG

// MARK: - Reusable RAG Commands

/// Search command - can be composed into any CLI
public struct RAGSearchCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rag-search",
        abstract: "Search for relevant content using RAG"
    )
    
    @Argument(help: "Search query")
    public var query: String
    
    @Option(name: .shortAndLong, help: "Maximum number of results")
    public var limit: Int = 5
    
    @Option(name: .long, help: "Number of candidates for reranking")
    public var candidates: Int = 100
    
    @Option(name: .long, help: "Database file path")
    public var database: String = RAGDefaults.databasePath()
    
    @Option(name: .long, help: "MLX model ID")
    public var model: String = RAGDefaults.modelId
    
    @Flag(name: .long, help: "Skip reranking (faster but less precise)")
    public var noRerank: Bool = false
    
    @Flag(name: .long, help: "Use Ollama backend instead of MLX")
    public var ollama: Bool = false
    
    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false
    
    public init() {}
    
    public func run() async throws {
        let engine: RAGEngine
        if ollama {
            engine = try RAGEngine(databasePath: database)
        } else {
            engine = try RAGEngine(databasePath: database, mlxModelId: model)
        }
        
        let results = try await engine.search(
            query: query,
            limit: limit,
            candidateCount: candidates,
            useReranker: !noRerank
        )
        
        if json {
            let data = try JSONEncoder().encode(results)
            print(String(data: data, encoding: .utf8)!)
        } else {
            if results.isEmpty {
                print("No results found.")
            } else {
                for (i, result) in results.enumerated() {
                    print("\n[\(i + 1)] \(result.id) (score: \(String(format: "%.2f", result.score)))")
                    print("    \(result.snippet)")
                }
            }
        }
    }
}

/// Fetch command - retrieve content by ID
public struct RAGFetchCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rag-fetch",
        abstract: "Fetch content by chunk or document ID"
    )
    
    @Argument(help: "Chunk or document ID")
    public var id: String
    
    @Option(name: .long, help: "Fetch mode: chunk, context, full")
    public var mode: String = "chunk"
    
    @Option(name: .long, help: "Context size (for context mode)")
    public var context: Int = 2
    
    @Option(name: .long, help: "Database file path")
    public var database: String = RAGDefaults.databasePath()
    
    public init() {}
    
    public func run() async throws {
        let engine = try RAGEngine(databasePath: database)
        
        let fetchMode: FetchMode
        switch mode {
        case "context": fetchMode = .context
        case "full": fetchMode = .full
        default: fetchMode = .chunk
        }
        
        let result = try await engine.fetch(id: id, mode: fetchMode, contextSize: context)
        print(result.content)
    }
}

/// Status command - check backend availability
public struct RAGStatusCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rag-status",
        abstract: "Check RAG backend status"
    )
    
    @Option(name: .long, help: "Database file path")
    public var database: String = RAGDefaults.databasePath()
    
    public init() {}
    
    public func run() async throws {
        let engine = try RAGEngine(databasePath: database)
        let status = await engine.checkBackend()
        
        print("RAG Status:")
        print("  Embedding: \(status.embedding ? "✅ Available" : "❌ Not available")")
        print("  Reranker:  \(status.reranker ? "✅ Available" : "❌ Not available")")
    }
}

// MARK: - Defaults

public enum RAGDefaults {
    public static let modelId = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    
    public static func databasePath() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".smith/rag/default.db").path
    }
}
