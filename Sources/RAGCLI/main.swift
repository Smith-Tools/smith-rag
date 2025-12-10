import ArgumentParser
import Foundation
import SmithRAG

@main
struct RAGCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rag",
        abstract: "Unified RAG search and fetch for Smith Tools",
        subcommands: [Search.self, Fetch.self, Status.self],
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
    
    @Flag(name: .long, help: "Skip reranking (faster but less precise)")
    var noRerank: Bool = false
    
    @Flag(name: .long, help: "Output as JSON")
    var json: Bool = false
    
    func run() async throws {
        let engine = try RAGEngine(databasePath: database)
        
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

// MARK: - Helpers

func defaultDatabasePath() -> String {
    let home = FileManager.default.homeDirectoryForCurrentUser
    return home.appendingPathComponent(".smith/rag/default.db").path
}
