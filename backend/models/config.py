from pydantic import BaseModel
from typing import Optional, Literal

class ModelConfig(BaseModel):
    llm_model: str
    embedding_model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_k: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store: Literal["chroma", "faiss"] = "chroma"
    
    class Config:
        json_schema_extra = {
            "example": {
                "llm_model": "gpt-3.5-turbo",
                "embedding_model": "text-embedding-ada-002",
                "temperature": 0.7,
                "max_tokens": 500,
                "top_k": 3,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "vector_store": "chroma"
            }
        } 