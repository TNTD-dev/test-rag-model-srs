import os
import chromadb
import PyPDF2
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import Data Models
from src.app.models import ChunkMetadata
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

class RAGIndexer:
    """
    Service chịu trách nhiệm:
    1. Đọc tài liệu (Loader)
    2. Chia nhỏ văn bản (Chunker)
    3. Tạo Vector Embeddings (dùng Gemini)
    4. Lưu vào ChromaDB
    """
    
    def __init__(self, persist_path: str = "./rag_db"):
        """
        Khởi tạo ChromaDB client và Embedding Function.
        
        Gợi ý implement:
        - Dùng chromadb.PersistentClient(path=persist_path)
        - Dùng embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=...)
        - Lấy hoặc tạo collection "srs_knowledge_base"
        """
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="srs_knowledge_base", 
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def load_documents(self, folder_path: str) -> List[Dict[str, Any]]:
        """Đọc tài liệu đa định dạng với xử lý lỗi."""
        documents = []
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist.")
            return documents

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                content = ""
                if filename.endswith(".md") or filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                elif filename.endswith(".pdf"):
                    # Senior Tip: Dùng thư viện chuyên dụng cho PDF
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                
                if content:
                    documents.append({
                        "content": content, 
                        "metadata": {"filename": filename, "path": file_path}
                    })
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                
        return documents

    def chunk_text(self, text: str, initial_metadata: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
        """
        Chia nhỏ văn bản thành các chunks có kích thước vừa phải (~500-1000 tokens).
        Quan trọng: Cố gắng giữ ngữ cảnh (Context) bằng cách chia theo Markdown Header (#, ##).
        """
        import uuid
        from datetime import datetime
        
        # Bước 1: Chia theo Markdown Header
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(text)

        # Bước 2: Chia nhỏ tiếp bằng RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        final_chunks = []
        
        # Nếu file không có markdown header nào, xử lý nguyên văn bản
        if not md_header_splits:
            from langchain_core.documents import Document
            md_header_splits = [Document(page_content=text, metadata={})]

        for split in md_header_splits:
            # Sub-split content của từng section
            sub_chunks = text_splitter.split_text(split.page_content)
            
            for chunk_content in sub_chunks:
                # Merge metadata: File metadata + Header metadata
                combined_metadata = {**initial_metadata, **split.metadata}
                
                # Bổ sung các trường bắt buộc theo ChunkMetadata Schema
                # Lưu ý: ChromaDB chỉ hỗ trợ flat metadata (str, int, float, bool), không nested dict
                full_metadata = {
                    "doc_id": str(uuid.uuid4()),
                    "source": combined_metadata.get("filename", "unknown"),
                    "section": combined_metadata.get("Header 1", combined_metadata.get("Header 2", "General")),
                    "category": "knowledge", # Default, logic phân loại có thể cải tiến sau
                    "created_at": datetime.now().isoformat(),
                    # Giữ lại các field khác nếu cần
                    **{k: v for k, v in combined_metadata.items() if k not in ["source", "section", "category", "created_at"]}
                }
                
                final_chunks.append((chunk_content, full_metadata))
                
        return final_chunks

    def build_index(self, folder_path: str):
        """Main flow chuyển sang chạy Local Embeddings (Nhanh hơn, không rate limit)."""
        print(f"Đang bắt đầu index dữ liệu từ: {folder_path}...")
        
        raw_docs = self.load_documents(folder_path)
        if not raw_docs:
            print("Không có tài liệu nào để xử lý.")
            return

        all_texts = []
        all_metadatas = []
        all_ids = []

        print(f"Đã đọc xong {len(raw_docs)} files. Bắt đầu chunking...")

        for doc in raw_docs:
            try:
                chunks = self.chunk_text(doc["content"], doc["metadata"])
                
                for i, (content, metadata) in enumerate(chunks):
                    all_texts.append(content)
                    all_metadatas.append(metadata)
                    all_ids.append(metadata["doc_id"]) # Dùng UUID từ bước chunk
            except Exception as e:
                print(f"Lỗi khi chunk file {doc['metadata'].get('filename')}: {e}")
                continue

        if not all_texts:
            print("Không có chunk nào được tạo ra.")
            return

        print(f"Tổng cộng: {len(all_texts)} chunks. Đang lưu vào ChromaDB...")

        # Với Model Local, ta có thể tăng batch size lên lớn hơn và không cần Retry
        batch_size = 50 
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_texts), batch_size):
            batch_end = i + batch_size
            current_batch = i // batch_size + 1
            print(f"Processing batch {current_batch}/{total_batches}...")
            
            try:
                self.collection.upsert(
                    documents=all_texts[i : batch_end],
                    metadatas=all_metadatas[i : batch_end],
                    ids=all_ids[i : batch_end]
                )
            except Exception as e:
                print(f"❌ Lỗi khi lưu batch {current_batch}: {e}")
            
        print(f"Hoàn thành! Đã index {len(all_texts)} chunks vào ChromaDB tại '{self.persist_path}'.")
