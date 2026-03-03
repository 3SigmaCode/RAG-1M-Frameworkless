import asyncio
import logging
import sys
import os
from typing import List, Dict, Any 
from dataclasses import dataclass 
import aiohttp
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.http import models as rest_models
import re
# Fix Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.vector_client import DualWriteDBClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]

class IngestionWorker:
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
        self.batch_buffer: List[DocumentChunk] = []
        
        logging.info("Loading BGE-Large (Dense) and SPLADE (Sparse) models into RAM...")
        # 1. The Net (Semantics)
        self.dense_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        # 2. The Sniper (Exact Keywords)
        self.sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
        
        self.db_client = DualWriteDBClient()

    async def setup(self):
        await self.db_client.connect()

    async def teardown(self):
        await self.db_client.close()

    async def extract_text(self, file_url: str) -> str:
        logging.info(f"Downloading stream from: {file_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(file_url, timeout=15) as response:
                    if response.status != 200:
                        logging.error(f"Failed to fetch {file_url}")
                        return ""
                    pdf_bytes = await response.read()
            except Exception as e:
                logging.error(f"Network error on {file_url}: {str(e)}")
                return ""

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_text = "".join([page.get_text() for page in doc])
        doc.close()
        
        # Brutal data sanitization
        return extracted_text.replace('\x00', '').replace('\u0000', '')
    
    def semantic_chunker(self, text: str, doc_id: str) -> List[DocumentChunk]:
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        
        chunk_size = 400
        overlap = 50
        processed_chunks = []
        
        # 2. Sliding Window Execution
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                break
                
            chunk_text = " ".join(chunk_words)
            processed_chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_index=len(processed_chunks),
                text=chunk_text,
                metadata={"source": "arxiv", "processed_at": "timestamp"}
            ))
            
        return processed_chunks
    
    async def process_document(self, file_url: str, doc_id: str):
        raw_text = await self.extract_text(file_url)
        chunks = self.semantic_chunker(raw_text, doc_id)
        for chunk in chunks:
            await self.queue.put(chunk)
    
    async def embed_and_upsert_worker(self):
        while True:
            chunk: DocumentChunk = await self.queue.get()
            self.batch_buffer.append(chunk)

            if len(self.batch_buffer) >= self.batch_size:
                await self._flush_batch()
            self.queue.task_done()
            
    def _generate_sparse_sync(self, texts: List[str]):
        """Helper to run the generator synchronously inside the thread."""
        return list(self.sparse_model.embed(texts))
    
    async def _flush_batch(self):
        if not self.batch_buffer:
            return
            
        logging.info(f"Flushing Hybrid batch of {len(self.batch_buffer)} chunks...")
        texts = [chunk.text for chunk in self.batch_buffer]
        
        # 1. Compute Dense Vectors (The Math of Meaning)
        dense_vectors = await asyncio.to_thread(
            self.dense_model.encode, texts, normalize_embeddings=True
        )
        dense_list = dense_vectors.tolist()
        
        # 2. Compute Sparse Vectors (The Math of Keywords)
        sparse_embeddings = await asyncio.to_thread(self._generate_sparse_sync, texts)
        
        # Format for Qdrant
        qdrant_sparse_vectors = []
        for sparse in sparse_embeddings:
            qdrant_sparse_vectors.append(
                rest_models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist()
                )
            )
        
        chunks_data = [
            {"doc_id": c.doc_id, "chunk_index": c.chunk_index, "text": c.text, "metadata": c.metadata} 
            for c in self.batch_buffer
        ]
        
        # 3. Execute Dual-Write
        await self.db_client.upsert_batch(chunks_data, dense_list, qdrant_sparse_vectors)
        self.batch_buffer.clear()

async def main():
    worker = IngestionWorker(batch_size=5) 
    await worker.setup()
    upsert_task = asyncio.create_task(worker.embed_and_upsert_worker())
    
    pdf_url_1 = "https://arxiv.org/pdf/1706.03762.pdf"  # Attention Is All You Need
    pdf_url_2 = "https://arxiv.org/pdf/2005.11401.pdf"  # RAG
    
    await asyncio.gather(
        worker.process_document(pdf_url_1, "arxiv_attention_paper"),
        worker.process_document(pdf_url_2, "arxiv_rag_paper")
    )

    await worker.queue.join()
    upsert_task.cancel()
    await worker.teardown()
    logging.info("Hybrid Ingestion Pipeline Execution Complete.")

if __name__ == "__main__":
    asyncio.run(main())