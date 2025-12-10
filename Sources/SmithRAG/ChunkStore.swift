import Foundation
import GRDB

/// Storage for document chunks with embeddings
public actor ChunkStore {
    private let dbQueue: DatabaseQueue
    
    public init(databasePath: String) throws {
        dbQueue = try DatabaseQueue(path: databasePath)
        try Self.migrate(dbQueue)
    }
    
    private static func migrate(_ dbQueue: DatabaseQueue) throws {
        try dbQueue.write { db in
            try db.create(table: "documents", ifNotExists: true) { t in
                t.column("id", .text).primaryKey()
                t.column("title", .text).notNull()
                t.column("url", .text)
                t.column("full_content", .text).notNull()
                t.column("created_at", .datetime).defaults(sql: "CURRENT_TIMESTAMP")
            }
            
            try db.create(table: "chunks", ifNotExists: true) { t in
                t.column("id", .text).primaryKey()
                t.column("document_id", .text).notNull()
                    .references("documents", onDelete: .cascade)
                t.column("chunk_index", .integer).notNull()
                t.column("text", .text).notNull()
                t.column("vector", .blob)
                t.uniqueKey(["document_id", "chunk_index"])
            }
            
            // FTS5 for fallback keyword search
            try db.execute(sql: """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
                USING fts5(text, content=chunks, content_rowid=rowid)
            """)
        }
    }
    
    // MARK: - Document Operations
    
    public func insertDocument(id: String, title: String, url: String?, content: String) throws {
        try dbQueue.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO documents (id, title, url, full_content) VALUES (?, ?, ?, ?)",
                arguments: [id, title, url, content]
            )
        }
    }
    
    public func fetchDocument(id: String) throws -> (title: String, content: String)? {
        try dbQueue.read { db in
            let row = try Row.fetchOne(
                db,
                sql: "SELECT title, full_content FROM documents WHERE id = ?",
                arguments: [id]
            )
            guard let row else { return nil }
            return (row["title"], row["full_content"])
        }
    }
    
    // MARK: - Chunk Operations
    
    public func insertChunk(id: String, documentId: String, index: Int, text: String, vector: [Float]?) throws {
        try dbQueue.write { db in
            let vectorBlob = vector.map { floatsToBlob($0) }
            try db.execute(
                sql: "INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, text, vector) VALUES (?, ?, ?, ?, ?)",
                arguments: [id, documentId, index, text, vectorBlob]
            )
            
            // Update FTS index
            try db.execute(
                sql: "INSERT INTO chunks_fts(rowid, text) SELECT rowid, text FROM chunks WHERE id = ?",
                arguments: [id]
            )
        }
    }
    
    public func fetchChunk(id: String) throws -> (text: String, documentId: String)? {
        try dbQueue.read { db in
            let row = try Row.fetchOne(
                db,
                sql: "SELECT text, document_id FROM chunks WHERE id = ?",
                arguments: [id]
            )
            guard let row else { return nil }
            return (row["text"], row["document_id"])
        }
    }
    
    public func fetchChunksWithContext(chunkId: String, contextSize: Int) throws -> [String] {
        try dbQueue.read { db in
            // Get the chunk's document and index
            guard let info = try Row.fetchOne(
                db,
                sql: "SELECT document_id, chunk_index FROM chunks WHERE id = ?",
                arguments: [chunkId]
            ) else { return [] }
            
            let documentId: String = info["document_id"]
            let chunkIndex: Int = info["chunk_index"]
            
            // Fetch surrounding chunks
            let rows = try Row.fetchAll(
                db,
                sql: """
                    SELECT text FROM chunks 
                    WHERE document_id = ? 
                    AND chunk_index BETWEEN ? AND ?
                    ORDER BY chunk_index
                """,
                arguments: [documentId, chunkIndex - contextSize, chunkIndex + contextSize]
            )
            
            return rows.map { $0["text"] as String }
        }
    }
    
    // MARK: - Vector Operations
    
    public func fetchAllVectors() throws -> [(id: String, vector: [Float])] {
        try dbQueue.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: "SELECT id, vector FROM chunks WHERE vector IS NOT NULL"
            )
            
            return rows.compactMap { row -> (String, [Float])? in
                guard let id: String = row["id"],
                      let blob: Data = row["vector"] else { return nil }
                return (id, blobToFloats(blob))
            }
        }
    }
    
    // MARK: - FTS Search (fallback)
    
    public func keywordSearch(query: String, limit: Int) throws -> [(id: String, snippet: String)] {
        try dbQueue.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: """
                    SELECT c.id, snippet(chunks_fts, 0, '', '', '...', 32) as snippet
                    FROM chunks_fts f
                    JOIN chunks c ON c.rowid = f.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """,
                arguments: [query, limit]
            )
            
            return rows.map { ($0["id"], $0["snippet"]) }
        }
    }
    
    // MARK: - Helpers
    
    private func floatsToBlob(_ floats: [Float]) -> Data {
        floats.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }
    
    private func blobToFloats(_ data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { ptr in
            Array(UnsafeBufferPointer(start: ptr.bindMemory(to: Float.self).baseAddress, count: count))
        }
    }
}
