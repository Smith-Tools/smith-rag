import XCTest
@testable import SmithRAG

final class SmithRAGTests: XCTestCase {
    func testVectorSearchDotProduct() {
        let search = VectorSearch()
        
        let candidates = [
            (id: "a", vector: [1.0, 0.0, 0.0] as [Float]),
            (id: "b", vector: [0.0, 1.0, 0.0] as [Float]),
            (id: "c", vector: [0.7, 0.7, 0.0] as [Float])
        ]
        
        let query: [Float] = [1.0, 0.0, 0.0]
        let results = search.search(query: query, candidates: candidates, topK: 2)
        
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].id, "a") // Exact match
        XCTAssertEqual(results[1].id, "c") // Partial match
    }
    
    func testNormalization() {
        let search = VectorSearch()
        let vector: [Float] = [3.0, 4.0]
        let normalized = search.normalize(vector)
        
        // 3-4-5 triangle, normalized to unit length
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.001)
    }
}
