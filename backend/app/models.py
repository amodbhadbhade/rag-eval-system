from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class DatasetInfo(BaseModel):
    """Information about a dataset"""

    dataset_id: str
    name: str


class DatasetListResponse(BaseModel):
    """Response for listing datasets"""

    datasets: List[DatasetInfo]
    total_count: int


class QuestionAnswer(BaseModel):
    """A question-answer pair"""

    question: str
    ground_truth: str


class DatasetQuestionsResponse(BaseModel):
    """Response containing questions and ground truth answers for a dataset"""

    dataset_id: str
    name: str
    questions_and_answers: List[QuestionAnswer]
    total_questions: int


class CreateDatasetResponse(BaseModel):
    """Response after creating a dataset"""

    dataset_id: str
    message: str


class EvaluationRequest(BaseModel):
    """Request payload for evaluating a response"""

    dataset_id: str
    question: str
    answer: str
    contexts: List[str]
    ragsource: Optional[str] = None
    pdfloader: Optional[str] = None
    hasimage: Optional[bool] = None
    hastable: Optional[bool] = None
    textsplitter: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None
    generator_llm: Optional[str] = None
    temperature: Optional[float] = None


class EvaluationScore(BaseModel):
    """Evaluation scores for a response"""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_similarity: float
    answer_correctness: float


class ErrorResponse(BaseModel):
    """Error response"""

    error: str
