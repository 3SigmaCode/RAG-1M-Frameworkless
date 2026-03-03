import asyncio
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridRetrievalEngine:
    def __init__(self):
        logging.info("Loading 3 Neural Networks into RAM. Brace your CPU...")
        # 1. Dense Semantic Model
        self.dense_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        # 2. Sparse Keyword Model (SPLADE)
        self.sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
        # 3. Cross-Encoder Sniper
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        
        self.qdrant = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
        self.collection_name = "land_records"

    def _generate_sparse_sync(self, text: str):
        return list(self.sparse_model.embed([text]))[0]

    async def search(self, query: str, initial_k: int = 15, final_k: int = 3):
        logging.info(f"Target Acquired: '{query}'")
        
        # 1. Generate both Dense and Sparse Vectors simultaneously
        dense_task = asyncio.to_thread(self.dense_model.encode, query, normalize_embeddings=True)
        sparse_task = asyncio.to_thread(self._generate_sparse_sync, query)
        
        dense_vector, sparse_vector = await asyncio.gather(dense_task, sparse_task)
        
        # Format sparse vector for Qdrant
        qdrant_sparse = rest_models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist()
        )

        # 2. Execute Hybrid Search using Reciprocal Rank Fusion (RRF)
        logging.info("Firing Dual-Payload at Qdrant and executing RRF Fusion...")
        raw_results = await self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                rest_models.Prefetch(
                    query=dense_vector.tolist(),
                    using="dense",
                    limit=initial_k,
                ),
                rest_models.Prefetch(
                    query=qdrant_sparse,
                    using="sparse",
                    limit=initial_k,
                )
            ],
            query=rest_models.FusionQuery(fusion=rest_models.Fusion.RRF),
            with_payload=True
        )

        if not raw_results.points:
            logging.warning("No relevant chunks found in the database.")
            return []

        # 3. The Sniper (Cross-Encoder Re-ranking)
        # 3. The Sniper (Cross-Encoder Re-ranking)
        logging.info(f"Stage 2: Re-ranking {len(raw_results.points)} fused chunks with Cross-Encoder...")
        pairs = [[query, hit.payload['text']] for hit in raw_results.points]
        cross_scores = await asyncio.to_thread(self.reranker.predict, pairs)
        
        # THE FIX: Extract into a native dictionary. Do not mutate the strict Pydantic model.
        reranked_results = []
        for i, hit in enumerate(raw_results.points):
            reranked_results.append({
                "doc_id": hit.payload['doc_id'],
                "text": hit.payload['text'],
                "cross_score": float(cross_scores[i]),
                "rrf_score": hit.score
            })
            
        # Sort by the brutal Cross-Encoder score
        reranked_results = sorted(reranked_results, key=lambda x: x["cross_score"], reverse=True)
        return reranked_results[:final_k]

async def main():
    engine = HybridRetrievalEngine()
    
    # We test the system with a highly specific technical query
    query = "What is the role of the multi-head attention mechanism?"
    
    results = await engine.search(query, initial_k=15, final_k=3)
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    # Updated to access dictionary keys instead of class attributes
    for i, hit in enumerate(results):
        print(f"\n[FINAL RANK {i+1}] Cross-Encoder Score: {hit['cross_score']:.4f} | RRF Fusion Score: {hit['rrf_score']:.4f}")
        print(f"Document: {hit['doc_id']}")
        print(f"Extracted Context: {hit['text'][:300]}...\n")

if __name__ == "__main__":
    asyncio.run(main())