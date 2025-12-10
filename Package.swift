// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "smith-rag",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SmithRAG", targets: ["SmithRAG"]),
        .executable(name: "rag", targets: ["RAGCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "6.24.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0")
    ],
    targets: [
        .target(
            name: "SmithRAG",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .executableTarget(
            name: "RAGCLI",
            dependencies: [
                "SmithRAG",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "SmithRAGTests",
            dependencies: ["SmithRAG"]
        )
    ]
)
