import ArgumentParser
import SmithRAGCommands

@main
struct RAGCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "smith-rag",
        abstract: "Unified RAG search and fetch for Smith Tools",
        subcommands: [
            RAGSearchCommand.self,
            RAGFetchCommand.self,
            RAGStatusCommand.self,
        ],
        defaultSubcommand: RAGSearchCommand.self
    )
}
