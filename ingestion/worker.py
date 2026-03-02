import asyncio
import logging
from typing import List, Dict, Any 
from dataclasses import dataclass 
import aiohttp
import fitz  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]  # Fixed from '=' to ':'

class IngestionWorker:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
        self.batch_buffer: List[DocumentChunk] = []

    async def extract_text(self, file_url: str) -> str:  # Fixed parameter name
        logging.info(f"Downloading stream from: {file_url}")
        
        # 1. Async network call to fetch the PDF bytes
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch {file_url}")
                    return ""
                pdf_bytes = await response.read()

        # 2. Parse the bytes in memory using PyMuPDF
        logging.info(f"Parsing PDF bytes in memory for {file_url}...")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text()
            
        doc.close()
        return extracted_text
    
    def semantic_chunker(self, text: str, doc_id: str) -> List[DocumentChunk]:
        raw_chunks = text.split("Section")  # Fixed from self.split
        processed_chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():  # Fixed typo chunk_txt
                continue
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_index=idx,
                text=f"Section {chunk_text.strip()}",
                metadata={
                    "source": "land_registry_db",
                    "processed_at": "timestamp"
                }
            )
            processed_chunks.append(chunk)
        return processed_chunks
    
    async def process_document(self, file_url: str, doc_id: str): # Fixed parameter name
        logging.info(f"Starting extraction for document: {doc_id}")
        raw_text = await self.extract_text(file_url)
        chunks = self.semantic_chunker(raw_text, doc_id)

        for chunk in chunks:
            await self.queue.put(chunk)
            logging.info(f"Queued chunk {chunk.chunk_index} for {doc_id}")
    
    async def embed_and_upsert_worker(self):
        while True:
            chunk: DocumentChunk = await self.queue.get()
            self.batch_buffer.append(chunk)

            if len(self.batch_buffer) >= self.batch_size:
                await self._flush_batch()
            self.queue.task_done()
    
    async def _flush_batch(self):  # Renamed for accuracy
        logging.info(f"Flushing batch of {len(self.batch_buffer)} chunks to Vector DB...")
        await asyncio.sleep(0.2)
        self.batch_buffer.clear()

async def main():
    worker = IngestionWorker(batch_size=2)

    upsert_task = asyncio.create_task(worker.embed_and_upsert_worker())
    
    # Test with two active PDFs
    pdf_url_1 = "https://arxiv.org/pdf/1706.03762.pdf"  # Attention Is All You Need
    pdf_url_2 = "https://arxiv.org/pdf/2005.11401.pdf"
    
    await asyncio.gather(
        worker.process_document(pdf_url_1, "remote_doc_1"),
        worker.process_document(pdf_url_2, "remote_doc_2")
    )

    await worker.queue.join()
    upsert_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())