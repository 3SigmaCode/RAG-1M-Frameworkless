import asyncio
import logging
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models
from typing import List,Dict,Any 



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DualWriteDBClient:
    def __init__(self):
        self.qdrant=AsyncQdrantClient(host="localhost")
        self.collection_name="land_records"
        self.pg_pool=None

    async def connect(self):
        logging.info("Establishing PostgreSQL connection pool...")
        self.pg_pool=await asyncpg.create_pool(
            user='ai_architect',
            password='rootpassword123',
            database='compliance_db',
            host='127.0.0.1',
            port=5432,
            min_size=5,
            max_size=20 # Handles high concurrency
        )
        await self._init_db_schema()

    async def _init_db_schema(self):
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
        
        collections=await self.qdrant.get_collections()
        if not any(c.name == self.collection_name for c in collections.collections):
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=1024, 
                    distance=rest_models.Distance.COSINE
                )
            )
            logging.info(f"Created Qdrant collection: {self.collection_name}")
    
    async def upsert_batch(self,chunks:List[Dict[str,Any]],vectors:List[List[float]]):
        import uuid
        import json


        points_ids = [str(uuid.uuid4()) for _ in chunks]

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    logging.info(f"Writing {len(chunks)} chunks to PostgreSQL...")
                    records_to_insert=[
                        (points_ids[i], c['doc_id'], c['chunk_index'], c['text'], json.dumps(c['metadata']))
                        for i, c in enumerate(chunks)
                    ]
                    await conn.executemany('''
                        INSERT INTO document_chunks (id, doc_id, chunk_index, text, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                    ''', records_to_insert)

                    logging.info(f"Writing {len(vectors)} vectors to Qdrant...")
                    await self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=rest_models.Batch(
                            ids=points_ids,
                            vectors=vectors,
                            payloads=chunks # We store the text as payload so Qdrant can return it instantly
                        )
                    )
                    logging.info("Dual-Write Transaction Successful.")
                    
                except Exception as e:
                    logging.error(f"Transaction Failed! Initiating Rollback. Error: {str(e)}")
                    # Because we are in an `async with conn.transaction():` block, 
                    # PostgreSQL will automatically roll back the inserts if Qdrant fails.
                    raise e


    async def close(self):
        await self.pg_pool.close()
        await self.qdrant.close()




# Add this to the very bottom of vector_client.py
import random

async def run_standalone_test():
    logging.info("--- STARTING DATABASE STRESS TEST ---")
    client = DualWriteDBClient()
    
    try:
        # 1. Boot up the connection pools
        await client.connect()
        
        # 2. Forge Synthetic Data (Mocking your Land Records)
        # We need fake text chunks and fake 1024-dimensional vectors
        dummy_chunks = [
            {"doc_id": "karnataka_plot_42", "chunk_index": 0, "text": "Ownership transferred to Shushant.", "metadata": {"source": "synthetic_test"}},
            {"doc_id": "karnataka_plot_42", "chunk_index": 1, "text": "Zoning regulations strictly commercial.", "metadata": {"source": "synthetic_test"}}
        ]
        
        # Generate two fake vectors, each with 1024 random floats
        logging.info("Generating synthetic 1024-D embedding matrices...")
        dummy_vectors = [[random.uniform(-1.0, 1.0) for _ in range(1024)] for _ in range(2)]
        
        # 3. Execute the SAGA Transaction
        await client.upsert_batch(dummy_chunks, dummy_vectors)
        
    except Exception as e:
        logging.error(f"Test Failed: {str(e)}")
    finally:
        # 4. Clean shutdown
        await client.close()
        logging.info("--- TEST COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(run_standalone_test())