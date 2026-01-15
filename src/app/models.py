from pydantic import BaseModel, Field
from typing import List, Optional

class SRSRequest(BaseModel):
    project_description: str = Field(..., description="Description of the project to generate SRS for")

class SRSResponse(BaseModel):
    srs_content: str = Field(..., description="Generated SRS content in Markdown format")
    rag_context: Optional[str] = None # Return applied context for transparency

class EvaluationRequest(BaseModel):
    srs_content: str = Field(..., description="SRS content to be evaluated")
    rag_context: Optional[str] = None # Pass context for Faithfulness check

class EvaluationScoreItem(BaseModel):
    raw: float
    weight: float
    weighted: float

class EvaluationScore(BaseModel):
    completeness: EvaluationScoreItem
    consistency: EvaluationScoreItem
    accuracy: EvaluationScoreItem
    format_tone: EvaluationScoreItem
    faithfulness: Optional[EvaluationScoreItem] = None # Optional for backward compatibility
    total_weighted_score: float

class EvaluationIssue(BaseModel):
    criterion: str
    problem: str
    evidence_quote: str
    impact: str
    recommendation: str

class EvaluationComment(BaseModel):
    summary: str
    strengths: List[str]
    issues: List[EvaluationIssue]
    quick_fixes: List[str]

class EvaluationResult(BaseModel):
    score: EvaluationScore
    group: str
    comment: EvaluationComment

class EvaluationResponse(BaseModel):
    evaluation_result: EvaluationResult = Field(..., description="Structured evaluation result")

class ChunkMetadata(BaseModel):
    doc_id: str = Field(..., description="Unique ID of the document")
    source: str = Field(..., description="Source filename")
    section: str = Field(..., description="Section name or header")
    category: str = Field(..., description="Category: 'rule', 'process', 'requirement', 'nfr'")
    created_at: str = Field(..., description="Indexing timestamp")

class RAGRequest(SRSRequest):
    use_rag: bool = Field(True, description="Whether to use RAG")

