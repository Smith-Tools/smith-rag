import ArgumentParser
import Foundation
import SmithRAG

@main
struct RAGCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rag",
        abstract: "Unified RAG search and fetch for Smith Tools",
        subcommands: [Search.self, Fetch.self, Status.self, Migrate.self],
        defaultSubcommand: Search.self
    )
}

// MARK: - Search Command

struct Search: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Search for relevant content across knowledge sources"
    )
    
    @Argument(help: "Search query")
    var query: String
    
    @Option(name: .shortAndLong, help: "Maximum number of results")
    var limit: Int = 5
    
    @Option(name: .long, help: "Number of candidates for reranking")
    var candidates: Int = 100
    
    @Option(name: .long, help: "Database file path")
    var database: String = defaultDatabasePath()
    
    @Option(name: .long, help: "MLX model ID")
    var model: String = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

    @Flag(name: .long, help: "Skip reranking (faster but less precise)")
    var noRerank: Bool = false
    
    @Flag(name: .long, help: "Use Ollama backend instead of MLX (legacy)")
    var ollama: Bool = false
    
    @Flag(name: .long, help: "Output as JSON")
    var json: Bool = false
    
    func run() async throws {
        let engine: RAGEngine
        if ollama {
            // Legacy Ollama backend
            engine = try RAGEngine(databasePath: database)
        } else {
            // MLX backend (default)
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

// MARK: - Fetch Command

struct Fetch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Fetch content by chunk or document ID"
    )
    
    @Argument(help: "Chunk or document ID")
    var id: String
    
    @Option(name: .long, help: "Fetch mode: chunk, context, full")
    var mode: String = "chunk"
    
    @Option(name: .long, help: "Context size (for context mode)")
    var context: Int = 2
    
    @Option(name: .long, help: "Database file path")
    var database: String = defaultDatabasePath()
    
    func run() async throws {
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

// MARK: - Status Command

struct Status: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Check Ollama status and model availability"
    )
    
    @Option(name: .long, help: "Database file path")
    var database: String = defaultDatabasePath()
    
    func run() async throws {
        let engine = try RAGEngine(databasePath: database)
        let status = await engine.checkOllama()
        
        print("Ollama Status:")
        print("  Embedding Model: \(status.embedding ? "✅ Available" : "❌ Not available")")
        print("  Reranker Model:  \(status.reranker ? "✅ Available" : "❌ Not available")")
        
        if !status.embedding {
            print("\nTo enable semantic search, run:")
            print("  ollama pull nomic-embed-text")
        }
        if !status.reranker {
            print("\nTo enable reranking, run:")
            print("  ollama pull bge-reranker-base")
        }
    }
}

// MARK: - Migrate Command

struct Migrate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Re-embed all chunks with MLX backend (migration from Ollama)"
    )
    
    @Option(name: .long, help: "Database file path")
    var database: String = defaultDatabasePath()
    
    @Option(name: .long, help: "MLX model ID")
    var model: String = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    
    @Flag(name: .long, help: "Confirm migration (required)")
    var confirm: Bool = false
    
    func run() async throws {
        guard confirm else {
            print("⚠️  Migration will re-embed all chunks with MLX.")
            print("   This will take 1-2 hours for ~14,000 chunks.")
            print("")
            print("   Current: 768d Ollama embeddings")
            print("   Target:  1024d MLX embeddings")
            print("")
            print("Run with --confirm to proceed:")
            print("  rag migrate --confirm")
            return
        }
        
        print("🚀 Starting MLX migration...")
        print("   Model: \(model)")
        print("   Database: \(database)")
        print("")
        
        let engine = try RAGEngine(databasePath: database, mlxModelId: model)
        
        let startTime = Date()
        
        let result = try await engine.reembedAll(batchSize: 64) { current, total in
            let elapsed = Date().timeIntervalSince(startTime)
            let rate = Double(current) / elapsed
            let remaining = Double(total - current) / rate
            
            let percent = Double(current) / Double(total) * 100
            print("\r⏳ Progress: \(current)/\(total) (\(String(format: "%.1f", percent))%) - ETA: \(formatTime(remaining))", terminator: "")
            fflush(stdout)
        }
        
        let totalTime = Date().timeIntervalSince(startTime)
        
        print("\n")
        print("✅ Migration complete!")
        print("   Success: \(result.success)")
        print("   Failed:  \(result.failed)")
        print("   Total:   \(result.total)")
        print("   Time:    \(formatTime(totalTime))")
    }
    
    private func formatTime(_ seconds: Double) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else if seconds < 3600 {
            return "\(Int(seconds / 60))m \(Int(seconds.truncatingRemainder(dividingBy: 60)))s"
        } else {
            let hours = Int(seconds / 3600)
            let mins = Int((seconds.truncatingRemainder(dividingBy: 3600)) / 60)
            return "\(hours)h \(mins)m"
        }
    }
}

// MARK: - Helpers

func defaultDatabasePath() -> String {
    let home = FileManager.default.homeDirectoryForCurrentUser
    return home.appendingPathComponent(".smith/rag/default.db").path
}

