import { ChromaClient, Collection } from "chromadb";
import { ChromaDBInstrumentation } from "@traceai/chromadb";
import {
  getCapturedSpans,
  clearCapturedSpans,
  findSpanByName,
  findSpansByPrefix,
  getSpanAttributes,
  generateRandomEmbedding,
  waitForService,
} from "./setup";

// Initialize instrumentation
const instrumentation = new ChromaDBInstrumentation();
instrumentation.enable();

const CHROMADB_URL = process.env.CHROMADB_URL || "http://localhost:8000";
const EMBEDDING_DIMENSIONS = 384;

describe("ChromaDB E2E Tests", () => {
  let client: ChromaClient;
  let collection: Collection;
  const testCollectionName = `test_collection_${Date.now()}`;

  beforeAll(async () => {
    client = new ChromaClient({ path: CHROMADB_URL });

    // Wait for ChromaDB to be ready
    await waitForService(async () => {
      try {
        await client.heartbeat();
        return true;
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    // Cleanup: delete test collection
    try {
      await client.deleteCollection({ name: testCollectionName });
    } catch {
      // Collection may not exist
    }
    instrumentation.disable();
  });

  beforeEach(() => {
    clearCapturedSpans();
  });

  describe("Collection Operations", () => {
    test("should trace collection creation", async () => {
      collection = await client.createCollection({
        name: testCollectionName,
        metadata: { description: "E2E test collection" },
      });

      expect(collection).toBeDefined();
      expect(collection.name).toBe(testCollectionName);

      // Verify trace was captured
      const spans = getCapturedSpans();
      expect(spans.length).toBeGreaterThan(0);
    });

    test("should trace get or create collection", async () => {
      const col = await client.getOrCreateCollection({
        name: testCollectionName,
      });

      expect(col).toBeDefined();
      const spans = getCapturedSpans();
      expect(spans.length).toBeGreaterThan(0);
    });
  });

  describe("Document Operations", () => {
    const testDocuments = [
      {
        id: "doc1",
        text: "Machine learning is a subset of artificial intelligence.",
        embedding: generateRandomEmbedding(EMBEDDING_DIMENSIONS),
      },
      {
        id: "doc2",
        text: "Deep learning uses neural networks with many layers.",
        embedding: generateRandomEmbedding(EMBEDDING_DIMENSIONS),
      },
      {
        id: "doc3",
        text: "Natural language processing enables computers to understand text.",
        embedding: generateRandomEmbedding(EMBEDDING_DIMENSIONS),
      },
    ];

    test("should trace document addition", async () => {
      await collection.add({
        ids: testDocuments.map((d) => d.id),
        documents: testDocuments.map((d) => d.text),
        embeddings: testDocuments.map((d) => d.embedding),
        metadatas: testDocuments.map((_, i) => ({ index: i })),
      });

      const spans = getCapturedSpans();
      const addSpan = spans.find((s) => s.name.includes("add"));

      if (addSpan) {
        const attrs = getSpanAttributes(addSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("add");
      }
    });

    test("should trace document query with semantic search", async () => {
      const queryEmbedding = generateRandomEmbedding(EMBEDDING_DIMENSIONS);

      const results = await collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: 2,
      });

      expect(results).toBeDefined();
      expect(results.ids).toBeDefined();
      expect(results.ids[0].length).toBeLessThanOrEqual(2);

      const spans = getCapturedSpans();
      const querySpan = spans.find((s) => s.name.includes("query"));

      if (querySpan) {
        const attrs = getSpanAttributes(querySpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("query");
        expect(attrs["db.vector.query.top_k"]).toBe(2);
      }
    });

    test("should trace document retrieval by ID", async () => {
      const results = await collection.get({
        ids: ["doc1", "doc2"],
      });

      expect(results).toBeDefined();
      expect(results.ids.length).toBe(2);

      const spans = getCapturedSpans();
      const getSpan = spans.find((s) => s.name.includes("get"));

      if (getSpan) {
        const attrs = getSpanAttributes(getSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("get");
      }
    });

    test("should trace document update", async () => {
      await collection.update({
        ids: ["doc1"],
        documents: ["Updated: Machine learning is a powerful AI technique."],
        metadatas: [{ index: 0, updated: true }],
      });

      const spans = getCapturedSpans();
      const updateSpan = spans.find((s) => s.name.includes("update"));

      if (updateSpan) {
        const attrs = getSpanAttributes(updateSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("update");
      }
    });

    test("should trace document upsert", async () => {
      await collection.upsert({
        ids: ["doc4"],
        documents: ["Reinforcement learning trains agents through rewards."],
        embeddings: [generateRandomEmbedding(EMBEDDING_DIMENSIONS)],
        metadatas: [{ index: 3 }],
      });

      const spans = getCapturedSpans();
      const upsertSpan = spans.find((s) => s.name.includes("upsert"));

      if (upsertSpan) {
        const attrs = getSpanAttributes(upsertSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("upsert");
      }
    });

    test("should trace collection count", async () => {
      const count = await collection.count();

      expect(count).toBeGreaterThan(0);

      const spans = getCapturedSpans();
      const countSpan = spans.find((s) => s.name.includes("count"));

      if (countSpan) {
        const attrs = getSpanAttributes(countSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("count");
      }
    });

    test("should trace peek operation", async () => {
      const results = await collection.peek({ limit: 2 });

      expect(results).toBeDefined();
      expect(results.ids.length).toBeLessThanOrEqual(2);

      const spans = getCapturedSpans();
      const peekSpan = spans.find((s) => s.name.includes("peek"));

      if (peekSpan) {
        const attrs = getSpanAttributes(peekSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("peek");
      }
    });

    test("should trace document deletion", async () => {
      await collection.delete({
        ids: ["doc4"],
      });

      const spans = getCapturedSpans();
      const deleteSpan = spans.find((s) => s.name.includes("delete"));

      if (deleteSpan) {
        const attrs = getSpanAttributes(deleteSpan);
        expect(attrs["db.system"]).toBe("chromadb");
        expect(attrs["db.operation.name"]).toBe("delete");
      }
    });
  });

  describe("Real-World Use Case: RAG Pipeline", () => {
    test("should trace complete RAG workflow", async () => {
      // Simulate a RAG (Retrieval-Augmented Generation) pipeline

      // Step 1: Ingest knowledge base documents
      const knowledgeBase = [
        {
          id: "kb_1",
          text: "Our company was founded in 2020 and specializes in AI solutions.",
          embedding: generateRandomEmbedding(EMBEDDING_DIMENSIONS),
        },
        {
          id: "kb_2",
          text: "We offer machine learning consulting and custom model development.",
          embedding: generateRandomEmbedding(EMBEDDING_DIMENSIONS),
        },
        {
          id: "kb_3",
          text: "Our team consists of PhD researchers and experienced engineers.",
          embedding: generateRandomEmbedding(EMBEDDING_DIMENSIONS),
        },
      ];

      await collection.add({
        ids: knowledgeBase.map((d) => d.id),
        documents: knowledgeBase.map((d) => d.text),
        embeddings: knowledgeBase.map((d) => d.embedding),
        metadatas: knowledgeBase.map((d) => ({ source: "knowledge_base" })),
      });

      // Step 2: User asks a question - simulate embedding the question
      const questionEmbedding = generateRandomEmbedding(EMBEDDING_DIMENSIONS);

      // Step 3: Retrieve relevant context
      const retrievalResults = await collection.query({
