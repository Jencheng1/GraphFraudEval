from typing import List, Dict, Any
import os
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from models.config import ModelConfig
import tempfile
import time

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.available_llm_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]
        self.available_embedding_models = [
            "text-embedding-ada-002",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]

    def get_available_llm_models(self) -> List[str]:
        return self.available_llm_models

    def get_available_embedding_models(self) -> List[str]:
        return self.available_embedding_models

    def _initialize_models(self, config: ModelConfig):
        # Initialize embedding model
        if config.embedding_model == "text-embedding-ada-002":
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model
            )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    async def process_document(self, file: UploadFile) -> Dict[str, Any]:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Load and process document
            loader = TextLoader(temp_file_path)
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)

            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings
            )

            return {
                "status": "success",
                "message": f"Processed {len(texts)} chunks from document",
                "num_chunks": len(texts)
            }

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    async def process_query(self, query: str, config: ModelConfig) -> Dict[str, Any]:
        start_time = time.time()
        
        # Initialize models if not already done
        if not self.llm or not self.embeddings:
            self._initialize_models(config)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": config.top_k}
            )
        )

        # Get response and retrieval context
        response = qa_chain.run(query)
        
        # Get retrieval context
        retrieval_context = self.vector_store.similarity_search(
            query,
            k=config.top_k
        )
        retrieval_context = [doc.page_content for doc in retrieval_context]
        
        # Calculate latency
        latency = time.time() - start_time

        return {
            "response": response,
            "retrieval_context": retrieval_context,
            "latency": latency,
            "model": config.llm_model,
            "embedding_model": config.embedding_model
        } 