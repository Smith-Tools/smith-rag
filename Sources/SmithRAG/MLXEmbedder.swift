import Foundation
import Logging
import MLX
import MLXEmbedders
import Tokenizers

/// Generates embeddings using Apple's official MLXEmbedders on Apple Silicon
public actor MLXEmbedder {
    private var container: MLXEmbedders.ModelContainer?
    private let modelId: String
    private let logger = Logger(label: "smith-rag.mlx-embedder")
    
    /// Initialize with model ID
    /// Supported models:
    /// - mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ (1024d, default)
    /// - nomic-ai/nomic-embed-text-v1.5 (768d)
    /// - BAAI/bge-small-en-v1.5 (384d)
    public init(modelId: String = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ") {
        self.modelId = modelId
    }
    
    /// Load the model (lazy initialization)
    private func ensureModelLoaded() async throws {
        guard container == nil else { return }
        
        logger.info("Loading MLX embedding model: \(modelId)")
        let startTime = Date()
        
        // Standard model configuration - relies on MLX/HuggingFace cache
        // Run `huggingface-cli download <model-id>` to ensure offline availability
        let config = ModelConfiguration(id: modelId)
        
        container = try await loadModelContainer(configuration: config) { progress in
            if progress.fractionCompleted < 1.0 {
                print("\rDownloading model: \(Int(progress.fractionCompleted * 100))%", terminator: "")
                fflush(stdout)
            }
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        logger.info("MLX model loaded in \(String(format: "%.2f", elapsed))s")
    }
    
    /// Generate embedding vector for text
    public func embed(_ text: String) async throws -> [Float] {
        try await ensureModelLoaded()
        
        guard let container = container else {
            throw MLXEmbedderError.modelNotLoaded
        }

        // Capture embedding dimension outside the Sendable closure
        let dim = embeddingDimension
        
        return await container.perform { model, tokenizer, pooling in
            // Tokenize input
            let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
            
            // Limit to max context
            let maxLength = min(tokens.count, 2048)
            let truncatedTokens = Array(tokens.prefix(maxLength))
            
            // Create input tensors
            let inputIds = MLXArray(truncatedTokens).reshaped([1, truncatedTokens.count])
            let padId = tokenizer.eosTokenId ?? 0
            let mask = (inputIds .!= padId)
            let tokenTypes = MLXArray.zeros(like: inputIds)
            
            // Generate embeddings - model returns EmbeddingModelOutput
            let output = model(
                inputIds,
                positionIds: nil,
                tokenTypeIds: tokenTypes,
                attentionMask: mask
            )
            
            // Apply pooling with normalization
            var pooledOutput = pooling(output, normalize: true, applyLayerNorm: true)

            // Some models ship without a pooling config (Pooler = .none). In that case the
            // pooling call returns [batch, seq, dim]; collapse across tokens to produce a
            // single embedding vector per input.
            let pooledShape = pooledOutput.shape.map { Int($0) }
            if pooledShape.count == 3, pooledShape[1] > 0 {
                pooledOutput = sum(pooledOutput, axis: 1) / MLXArray(Float(pooledShape[1]))
            }
            
            // Ensure computation is complete
            eval(pooledOutput)
            
            // Convert to Float array - take first item in batch
            let vector: [Float]
            let finalShape = pooledOutput.shape.map { Int($0) }
            if finalShape.count == 2, finalShape[0] > 0 {
                vector = pooledOutput[0, 0...].asArray(Float.self)
            } else {
                vector = pooledOutput.asArray(Float.self)
            }

            // Truncate in case extra dims slipped through (defensive)
            return vector.count > dim ? Array(vector.prefix(dim)) : vector
        }
    }
    
    /// Batch embed multiple texts
    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        try await ensureModelLoaded()
        
        guard let container = container else {
            throw MLXEmbedderError.modelNotLoaded
        }

        let dim = embeddingDimension
        
        return await container.perform { model, tokenizer, pooling in
            let inputs = texts.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            
            // Find max length, cap at 2048 (balanced for performance/memory)
            let maxLength = min(inputs.reduce(into: 16) { acc, elem in
                acc = max(acc, elem.count)
            }, 2048)
            
            // Pad tokens
            let padId = tokenizer.eosTokenId ?? 0
            let padded = stacked(
                inputs.map { elem in
                    let truncated = Array(elem.prefix(maxLength))
                    let padding = Array(repeating: padId, count: maxLength - truncated.count)
                    return MLXArray(truncated + padding)
                }
            )
            
            let mask = (padded .!= padId)
            let tokenTypes = MLXArray.zeros(like: padded)
            
            // Generate embeddings
            let output = model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)
            
            // Apply pooling with normalization
            var pooledOutput = pooling(output, normalize: true, applyLayerNorm: true)

            // Fallback pooling if none provided: reduce token dimension
            let pooledShape = pooledOutput.shape.map { Int($0) }
            if pooledShape.count == 3, pooledShape[1] > 0 {
                pooledOutput = sum(pooledOutput, axis: 1) / MLXArray(Float(pooledShape[1]))
            }
            
            // Ensure computation is complete
            eval(pooledOutput)
            
            // Convert each embedding to Float array (first item per batch row)
            return pooledOutput.map {
                let shape = $0.shape.map { Int($0) }
                let vector: [Float]
                if shape.count == 2, shape[0] > 0 {
                    vector = $0[0, 0...].asArray(Float.self)
                } else {
                    vector = $0.asArray(Float.self)
                }
                return vector.count > dim ? Array(vector.prefix(dim)) : vector
            }
        }
    }
    
    /// Get embedding dimension based on model
    public var embeddingDimension: Int {
        switch modelId {
        case let id where id.contains("Qwen3"):
            return 1024
        case let id where id.contains("nomic"):
            return 768
        case let id where id.contains("bge-small"):
            return 384
        case let id where id.contains("bge-base"):
            return 768
        case let id where id.contains("bge-large"):
            return 1024
        default:
            return 768
        }
    }
    
    /// Check if model is available
    public func isAvailable() async -> Bool {
        do {
            try await ensureModelLoaded()
            return true
        } catch {
            logger.error("MLX model not available: \(error)")
            return false
        }
    }
}

public enum MLXEmbedderError: Error, LocalizedError {
    case modelNotLoaded
    case embeddingFailed
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "MLX embedding model not loaded"
        case .embeddingFailed:
            return "Failed to generate embedding"
        }
    }
}
