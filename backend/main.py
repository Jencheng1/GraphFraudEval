from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from services.rag_service import RAGService
from services.evaluation_service import EvaluationService
from models.config import ModelConfig

app = FastAPI(title="LLM Evaluation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()
evaluation_service = EvaluationService()

class QueryRequest(BaseModel):
    query: str
    model_config: ModelConfig

class EvaluationRequest(BaseModel):
    queries: List[str]
    model_configs: List[ModelConfig]

@app.post("/api/query")
async def process_query(request: QueryRequest):
    try:
        response = await rag_service.process_query(request.query, request.model_config)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_models(request: EvaluationRequest):
    try:
        results = await evaluation_service.evaluate_models(
            request.queries,
            request.model_configs
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_documents(file: UploadFile = File(...)):
    try:
        result = await rag_service.process_document(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_available_models():
    return {
        "llm_models": rag_service.get_available_llm_models(),
        "embedding_models": rag_service.get_available_embedding_models()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 