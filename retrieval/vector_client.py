import asyncio
import logging
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DualWriteDBClient:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
        self.collection_name = "land_records"
        self.pg_pool = None

    async def connect(self):
        logging.info("Establishing PostgreSQL connection pool...")
        self.pg_pool = await asyncpg.create_pool(
            user='ai_architect',
            password='rootpassword123',
            database='compliance_db',
            host='127.0.0.1',
            port=5432,
            min_size=5,
            max_size=20
        )
        await self._init_db_schema()

    async def _init_db_schema(self):
        # 1. PostgreSQL Schema (Unchanged)
        async with self.pg_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB
                )
            ''')
        
        # 2. THE TEARDOWN: We use recreate_collection to wipe the old single-vector DB.
        # We now define a Hybrid Schema with "Named Vectors".
        logging.info("Wiping old Qdrant collection and initializing Hybrid Search Schema...")
        await self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": rest_models.VectorParams(
                    size=1024, 
                    distance=rest_models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": rest_models.SparseVectorParams()
            }
        )
        logging.info(f"Hybrid Qdrant collection ready: {self.collection_name}")

    async def upsert_batch(
        self, 
        chunks: List[Dict[str, Any]], 
        dense_vectors: List[List[float]], 
        sparse_vectors: List[rest_models.SparseVector] # The new payload
    ):
        import uuid
        import json
        
        points_ids = [str(uuid.uuid4()) for _ in chunks]
        
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # STEP 1: PostgreSQL
                    records_to_insert = [
                        (points_ids[i], c['doc_id'], c['chunk_index'], c['text'], json.dumps(c['metadata']))
                        for i, c in enumerate(chunks)
                    ]
                    
                    await conn.executemany('''
                        INSERT INTO document_chunks (id, doc_id, chunk_index, text, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                    ''', records_to_insert)

                    # STEP 2: Qdrant Hybrid Upsert
                    points = []
                    for i in range(len(chunks)):
                        points.append(
                            rest_models.PointStruct(
                                id=points_ids[i],
                                # We now pass a dictionary of vectors instead of a flat array
                                vector={
                                    "dense": dense_vectors[i],
                                    "sparse": sparse_vectors[i]
                                },
                                payload=chunks[i]
                            )
                        )

                    await self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logging.info(f"Hybrid SAGA Transaction Successful: {len(chunks)} chunks locked in.")
                    
                except Exception as e:
                    logging.error(f"Transaction Failed! Initiating Rollback. Error: {str(e)}")
                    raise e

    async def close(self):
        await self.pg_pool.close()
        await self.qdrant.close()

# Keep your existing standalone test block at the bottom, we will update it later.