from fastapi import FastAPI, HTTPException
from app.models import SRSRequest, SRSResponse, EvaluationRequest, EvaluationResponse
from app.services.srs_generator import SRSGenerator
from app.services.evaluator import Evaluator

app = FastAPI(
    title="SRS Generation API",
    description="API for generating Software Requirements Specification (SRS) documents using AI.",
    version="1.0.0"
)

# Initialize the service
srs_generator = SRSGenerator()
evaluator = Evaluator()

from src.app.models import RAGRequest, SRSResponse, EvaluationRequest, EvaluationResponse

# ... (Previous imports)

@app.post("/generate-srs", response_model=SRSResponse)
async def generate_srs(request: RAGRequest):
    """
    Generate an SRS document based on the provided project description.
    Supports RAG mode (use_rag=True/False).
    """
    try:
        srs_content, rag_context = srs_generator.generate_srs(request.project_description, use_rag=request.use_rag)
        return SRSResponse(srs_content=srs_content, rag_context=rag_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-srs", response_model=EvaluationResponse)
async def evaluate_srs(request: EvaluationRequest):
    """
    Evaluate an SRS document based on stringent criteria.
    Now supports RAG Faithfulness check if context is provided.
    """
    try:
        evaluation_result = evaluator.evaluate_srs(request.srs_content, request.rag_context)
        return EvaluationResponse(evaluation_result=evaluation_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve_docs(query: str, top_k: int = 5):
    """
    Debug endpoint for RAG Retrieval.
    """
    try:
        results = srs_generator.retriever.retrieve(query, top_k=top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}
