// swift-tools-version: 6.0
import PackageDescription

let package = Package(
	name: "smith-rag",
	platforms: [
		.macOS(.v15)
	],
	products: [
		.library(name: "SmithRAG", targets: ["SmithRAG"]),
		.library(name: "SmithRAGCommands", targets: ["SmithRAGCommands"]),
		.executable(name: "smith-rag", targets: ["RAGCLI"]),
	],
	dependencies: [
		.package(url: "https://github.com/groue/GRDB.swift.git", from: "6.24.0"),
		.package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
		.package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
		// Apple's official MLX Swift LM for embeddings
		.package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.29.1"),
	],
	targets: [
		.target(
			name: "SmithRAG",
			dependencies: [
				.product(name: "GRDB", package: "GRDB.swift"),
				.product(name: "Logging", package: "swift-log"),
				.product(name: "MLXEmbedders", package: "mlx-swift-lm"),
			]
		),
		.target(
			name: "SmithRAGCommands",
			dependencies: [
				"SmithRAG",
				.product(name: "ArgumentParser", package: "swift-argument-parser"),
			]
		),
		.executableTarget(
			name: "RAGCLI",
			dependencies: [
				"SmithRAGCommands",
				.product(name: "ArgumentParser", package: "swift-argument-parser"),
			],
			swiftSettings: [
				.unsafeFlags(["-parse-as-library"])
			]
		),
		.testTarget(
			name: "SmithRAGTests",
			dependencies: ["SmithRAG"]
		),
	]
)
