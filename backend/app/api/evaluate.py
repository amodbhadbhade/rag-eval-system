from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request
from app.services.evaluator import (
    evaluate_response,
    evaluate_reference_free,
    evaluate_response_deepeval,
    evaluate_reference_free_deepeval,
)
from app.db.mongo import save_evaluation, get_evaluation_history
from app.models import EvaluationRequest, EvaluationScore, ErrorResponse

router = APIRouter()


class ReferenceFreeEvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    expected_metrics: Optional[Dict[str, float]] = None
    dataset_id: Optional[str] = None
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


class RecommendationRequest(BaseModel):
    worst_metric: str
    worst_score: float
    threshold: float = 0.7


@router.post(
    "/",
    response_model=dict,
    summary="Evaluate a response against dataset",
    description="Evaluate an answer using RAGAS metrics against ground truth from a dataset",
)
async def evaluate(payload: EvaluationRequest, request: Request):
    """
    Evaluate a response using RAGAS metrics.

    This endpoint evaluates an answer against the ground truth from a dataset using multiple metrics:
    - **Faithfulness**: How faithful is the answer to the provided context
    - **Answer Relevancy**: How relevant is the answer to the question
    - **Context Precision**: Precision of the retrieved context
    - **Context Recall**: Recall of the retrieved context
    - **Answer Similarity**: Semantic similarity between generated and ground truth answers
    - **Answer Correctness**: Overall correctness of the answer

    Request body:
    - **dataset_id**: ID of the dataset containing ground truth
    - **question**: The question being answered
    - **answer**: The generated answer
    - **contexts**: List of context passages used to generate the answer

    Returns evaluation scores for each metric.
    """
    payload_dict = payload.dict()
    payload_dict["api_url"] = str(request.url)
    result = await evaluate_response(payload_dict)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    save_evaluation(payload_dict, result, "RAGAS")

    return result


@router.post(
    "/reference-free",
    response_model=dict,
    summary="Evaluate a response without ground truth",
    description="Evaluate an answer using reference-free RAGAS metrics (Faithfulness and Answer Relevancy).",
)
async def evaluate_no_gt(payload: ReferenceFreeEvaluationRequest, request: Request):
    """
    Evaluate a response using metrics that do not require a ground truth.

    Uses:
    - **Faithfulness**: Measures if the answer is inferred from the context.
    - **Answer Relevancy**: Measures how relevant the answer is to the question.
    """
    payload_dict = payload.dict()
    payload_dict["api_url"] = str(request.url)
    actual_metrics = await evaluate_reference_free(payload_dict)

    response = payload_dict.copy()
    response["actual_metrics"] = actual_metrics

    save_evaluation(payload_dict, actual_metrics, "RAGAS_ReferenceFree")

    return response


@router.post(
    "/deepeval/",
    response_model=dict,
    summary="Evaluate a response against dataset using DeepEval",
    description="Evaluate an answer using DeepEval metrics against ground truth from a dataset",
)
async def evaluate_with_deepeval(payload: EvaluationRequest, request: Request):
    """
    Evaluate a response using DeepEval metrics.

    Returns evaluation scores for:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall
    """
    payload_dict = payload.dict()
    payload_dict["api_url"] = str(request.url)
    result = await evaluate_response_deepeval(payload_dict)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    save_evaluation(payload_dict, result, "DeepEval")

    return result


@router.post(
    "/deepeval/reference-free",
    response_model=dict,
    summary="Evaluate a response without ground truth using DeepEval",
    description="Evaluate an answer using reference-free DeepEval metrics.",
)
async def evaluate_no_gt_deepeval(
    payload: ReferenceFreeEvaluationRequest, request: Request
):
    """
    Evaluate a response using metrics that do not require a ground truth (via DeepEval).

    Uses:
    - **Faithfulness**
    - **Answer Relevancy**
    """
    payload_dict = payload.dict()
    payload_dict["api_url"] = str(request.url)
    actual_metrics = await evaluate_reference_free_deepeval(payload_dict)

    response = payload_dict.copy()
    response["actual_metrics"] = actual_metrics

    save_evaluation(payload_dict, actual_metrics, "DeepEval_ReferenceFree")

    return response


@router.get(
    "/history",
    response_model=List[dict],
    summary="Get evaluation history",
    description="Fetch all historical evaluation records from the database.",
)
def get_history():
    """Fetch all evaluation records from MongoDB."""
    records = get_evaluation_history()
    return records


@router.post(
    "/recommendations",
    response_model=dict,
    summary="Get recommendations for RAG system tweaks",
    description="Fetch dynamic system tuning recommendations using the evaluator LLM or fallback defaults.",
)
async def get_system_recommendations(payload: RecommendationRequest):
    import os
    from app.services.evaluator import get_llm_recommendation

    get_llm_recs = os.getenv("GET_RECOMMENDATIONS_FROM_LLM", "False").lower() == "true"

    if get_llm_recs:
        try:
            recommendation = await get_llm_recommendation(
                payload.worst_metric, payload.worst_score, payload.threshold
            )
            return {"recommendation": recommendation, "source": "llm"}
        except Exception as e:
            return {"error": str(e), "source": "llm"}
    else:
        worst_metric = payload.worst_metric
        rec = ""
        if worst_metric == "faithfulness":
            rec = "**Diagnosis:** LLM Hallucination. The model is inventing facts beyond the retrieved context.\n\n**Fixes:** \n- Add strict grounding prompts ('Answer ONLY using context'). \n- Reduce LLM Temperature. \n- Implement context citation enforcement."
        elif worst_metric == "context_recall":
            rec = "**Diagnosis:** Missing Information. The retriever isn't finding the correct ground-truth passages.\n\n**Fixes:** \n- Increase Top-K retrieval. \n- Switch to a better embedding model. \n- Add hybrid search (BM25 + Vector)."
        elif worst_metric == "context_precision":
            rec = "**Diagnosis:** Noisy Context. The retriever is fetching irrelevant documents alongside good ones.\n\n**Fixes:** \n- Reduce Top-K. \n- Add a Re-ranker (e.g., Cohere/Cross-Encoder). \n- Improve document chunk granularity."
        elif worst_metric == "answer_relevancy":
            rec = "**Diagnosis:** Query Understanding Failure. The answer doesn't address the user's intent.\n\n**Fixes:** \n- Implement Query Rewriting before retrieval. \n- Tune system prompts to directly address the user."
        else:
            rec = "**Diagnosis:** Pipeline Optimization Needed.\n\n**Fixes:** \n- Review pipeline configurations."

        return {"recommendation": rec, "source": "static"}
