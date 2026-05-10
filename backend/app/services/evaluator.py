import os
from dotenv import load_dotenv

load_dotenv()
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)
from app.db.mongo import get_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


class EvaluatorChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI wrapper to enforce a system prompt for RAGAS evaluation."""

    system_prompt: str = "You are an expert, impartial evaluator of RAG systems. Follow the instructions strictly."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class EvaluatorChatOllama(ChatOllama):
    """Custom ChatOllama wrapper to enforce a system prompt for RAGAS evaluation."""

    system_prompt: str = "You are an expert, impartial evaluator of RAG systems. Follow the instructions strictly."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class DeepEvalOllamaModel(DeepEvalBaseLLM):
    """Custom DeepEval LLM wrapper for Ollama models."""

    model_name: str

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def load_model(self):
        return ChatOllama(model=self.model_name, temperature=0.0)

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.load_model().ainvoke(prompt)).content

    def get_model_name(self):
        return self.model_name


async def evaluate_response(payload):
    """
    Evaluate a response using RAGAS metrics.

    Args:
        payload: Dictionary containing:
            - dataset_id: ID of the dataset
            - question: The question being answered
            - answer: The generated answer
            - contexts: List of context passages

    Returns:
        Dictionary with evaluation scores or error message
    """
    dataset = get_dataset(payload["dataset_id"])
    if not dataset:
        return {"error": "Dataset not found"}

    data = dataset["data"]

    gt_answer = None
    for i, q in enumerate(data["question"]):
        if q == payload["question"]:
            gt_answer = data["ground_truth"][i]
            break

    if not gt_answer:
        return {"error": "Question not found in dataset"}

    eval_data = Dataset.from_dict(
        {
            "question": [payload["question"]],
            "answer": [payload["answer"]],
            "contexts": [payload["contexts"]],
            "ground_truth": [gt_answer],
        }
    )

    evaluator_model = os.getenv("EVALUATOR_LLM", "gpt-4o")
    is_ollama = os.getenv("OLLAMA_MODE", "False").lower() == "true"

    system_prompt = "You are a strict and expert RAG evaluator. Always follow the specific formatting instructions precisely, be completely unbiased, and ensure your outputs are perfectly formatted JSON where required."

    payload["evaluator_temperature"] = 0.0
    payload["evaluator_system_prompt"] = system_prompt

    if is_ollama:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]:
            evaluator_model = "llama3"  # Fallback Ollama model
        payload["evaluator_model"] = evaluator_model
        evaluator_llm = EvaluatorChatOllama(
            model=evaluator_model,
            temperature=0.0,
            system_prompt=system_prompt,
        )
        embeddings = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        )
    else:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo"]:
            evaluator_model = "gpt-4o"
        payload["evaluator_model"] = evaluator_model
        evaluator_llm = EvaluatorChatOpenAI(
            model=evaluator_model,
            temperature=0.0,
            system_prompt=system_prompt,
        )
        embeddings = OpenAIEmbeddings()

    score = evaluate(
        eval_data,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness,
        ],
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    # Convert numpy data types from the Ragas Result object to native Python floats
    return {key: float(value) for key, value in score.items()}


async def evaluate_reference_free(payload):
    """
    Evaluate a response using RAGAS metrics that do not require ground truth.

    Args:
        payload: Dictionary containing:
            - question: The question being answered
            - answer: The generated answer
            - contexts: List of context passages
    """
    eval_data = Dataset.from_dict(
        {
            "question": [payload["question"]],
            "answer": [payload["answer"]],
            "contexts": [payload["contexts"]],
        }
    )

    evaluator_model = os.getenv("EVALUATOR_LLM", "gpt-4o")
    is_ollama = os.getenv("OLLAMA_MODE", "False").lower() == "true"

    system_prompt = "You are a strict and expert RAG evaluator. Always follow the specific formatting instructions precisely, be completely unbiased, and ensure your outputs are perfectly formatted JSON where required."

    payload["evaluator_temperature"] = 0.0
    payload["evaluator_system_prompt"] = system_prompt

    if is_ollama:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]:
            evaluator_model = "llama3"
        payload["evaluator_model"] = evaluator_model
        evaluator_llm = EvaluatorChatOllama(
            model=evaluator_model,
            temperature=0.0,
            system_prompt=system_prompt,
        )
        embeddings = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        )
    else:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo"]:
            evaluator_model = "gpt-4o"
        payload["evaluator_model"] = evaluator_model
        evaluator_llm = EvaluatorChatOpenAI(
            model=evaluator_model,
            temperature=0.0,
            system_prompt=system_prompt,
        )
        embeddings = OpenAIEmbeddings()

    score = evaluate(
        eval_data,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    return {key: float(value) for key, value in score.items()}


async def evaluate_response_deepeval(payload):
    """
    Evaluate a response using DeepEval metrics against a ground truth dataset.
    """
    dataset = get_dataset(payload["dataset_id"])
    if not dataset:
        return {"error": "Dataset not found"}

    data = dataset["data"]

    gt_answer = None
    for i, q in enumerate(data["question"]):
        if q == payload["question"]:
            gt_answer = data["ground_truth"][i]
            break

    if not gt_answer:
        return {"error": "Question not found in dataset"}

    test_case = LLMTestCase(
        input=payload["question"],
        actual_output=payload["answer"],
        expected_output=gt_answer,
        retrieval_context=payload["contexts"],
    )

    evaluator_model = os.getenv("EVALUATOR_LLM", "gpt-4o")
    is_ollama = os.getenv("OLLAMA_MODE", "False").lower() == "true"

    if is_ollama:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]:
            evaluator_model = "llama3"
        eval_model = DeepEvalOllamaModel(model_name=evaluator_model)
    else:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo"]:
            evaluator_model = "gpt-4o"
        eval_model = evaluator_model

    payload["evaluator_model"] = evaluator_model
    payload["evaluator_temperature"] = 0.0
    payload["evaluator_system_prompt"] = "DeepEval internal prompts (metric-specific)"

    metrics_to_run = {
        "faithfulness": FaithfulnessMetric(
            threshold=0.0, model=eval_model, include_reason=False
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.0, model=eval_model, include_reason=False
        ),
        "context_precision": ContextualPrecisionMetric(
            threshold=0.0, model=eval_model, include_reason=False
        ),
        "context_recall": ContextualRecallMetric(
            threshold=0.0, model=eval_model, include_reason=False
        ),
    }

    scores = {}
    for name, metric in metrics_to_run.items():
        try:
            await metric.a_measure(test_case)
            scores[name] = float(metric.score)
        except Exception as e:
            print(f"DeepEval error measuring {name}: {e}")
            scores[name] = 0.0

    return scores


async def evaluate_reference_free_deepeval(payload):
    """
    Evaluate a response using DeepEval reference-free metrics.
    """
    # Ground truth (expected_output) is excluded for reference-free evaluation
    test_case = LLMTestCase(
        input=payload["question"],
        actual_output=payload["answer"],
        retrieval_context=payload["contexts"],
    )

    evaluator_model = os.getenv("EVALUATOR_LLM", "gpt-4o")
    is_ollama = os.getenv("OLLAMA_MODE", "False").lower() == "true"

    if is_ollama:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]:
            evaluator_model = "llama3"
        eval_model = DeepEvalOllamaModel(model_name=evaluator_model)
    else:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo"]:
            evaluator_model = "gpt-4o"
        eval_model = evaluator_model

    payload["evaluator_model"] = evaluator_model
    payload["evaluator_temperature"] = 0.0
    payload["evaluator_system_prompt"] = "DeepEval internal prompts (metric-specific)"

    # DeepEval handles reference-free checks well with these two metrics
    metrics_to_run = {
        "faithfulness": FaithfulnessMetric(
            threshold=0.0, model=eval_model, include_reason=False
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.0, model=eval_model, include_reason=False
        ),
    }

    scores = {}
    for name, metric in metrics_to_run.items():
        try:
            await metric.a_measure(test_case)
            scores[name] = float(metric.score)
        except Exception as e:
            print(f"DeepEval error measuring {name}: {e}")
            scores[name] = 0.0

    return scores


async def get_llm_recommendation(
    worst_metric: str, worst_score: float, threshold: float
) -> str:
    """
    Fetches actionable RAG system tweaks from the LLM based on failing metric scores.
    """
    evaluator_model = os.getenv("EVALUATOR_LLM", "gpt-4o")
    is_ollama = os.getenv("OLLAMA_MODE", "False").lower() == "true"
    system_prompt = """You are an expert AI evaluator for Retrieval-Augmented Generation (RAG) systems.

Your task is to analyze the weakest-performing RAG evaluation metric and generate actionable recommendations to improve system performance.

You will receive an input payload containing:
1. worst_metric → One of:
   - "faithfulness"
   - "answer_relevancy"
   - "context_precision"
   - "context_recall"

2. score → Current score of the metric

3. threshold → Expected minimum acceptable threshold

Your responsibilities:
- Analyze why the metric may be underperforming
- Explain what the metric means
- Identify likely root causes
- Provide detailed recommendations
- Suggest practical fixes across:
  - Retrieval pipeline
  - Chunking strategy
  - Embedding quality
  - Prompt engineering
  - Re-ranking
  - Context construction
  - LLM generation settings
  - Metadata filtering
  - Evaluation methodology
- Prioritize recommendations from highest impact to lowest impact
- Keep the response technically accurate and implementation-oriented

IMPORTANT RULES:
- Do NOT hallucinate missing information
- Do NOT assume any specific vector database or framework unless explicitly mentioned
- Be concise but detailed
- Recommendations must be actionable
- Use bullet points and structured formatting
- Mention expected impact of each recommendation
- Include warnings about tradeoffs where relevant
- If the score is close to threshold, suggest optimization-level fixes
- If the score is far below threshold, suggest architectural or pipeline-level fixes

Interpretation Guidelines:
- score >= threshold:
  Mention that the metric is already acceptable but can still be optimized.

- threshold - score <= 0.1:
  Treat as moderate degradation.

- threshold - score > 0.1:
  Treat as severe degradation.

Metric Definitions:

1. faithfulness
Measures whether the generated answer is factually grounded in the retrieved context and does not hallucinate unsupported claims.

Common causes of low faithfulness:
- Hallucinations from the LLM
- Irrelevant or noisy context
- Missing grounding instructions
- Excessively large context windows
- Weak prompt constraints

Typical fixes:
- Strong grounding prompts
- Citation enforcement
- Reduce irrelevant chunks
- Use re-ranking
- Lower generation temperature
- Context compression

2. answer_relevancy
Measures how well the generated answer addresses the user query.

Common causes:
- Poor query understanding
- Weak retrieval-query alignment
- Generic answers
- Prompt ambiguity
- Missing semantic similarity

Typical fixes:
- Improve embeddings
- Query rewriting
- Hybrid retrieval
- Better prompt instructions
- Intent-aware retrieval

3. context_precision
Measures how much of the retrieved context is actually relevant.

Common causes:
- Too many irrelevant chunks
- Large top-k retrieval
- Weak similarity search
- Poor chunking
- Missing metadata filters

Typical fixes:
- Reduce top-k
- Better chunk segmentation
- Re-ranking
- Metadata filtering
- Hybrid retrieval
- Semantic chunking

4. context_recall
Measures whether all necessary information was successfully retrieved.

Common causes:
- Missing documents
- Small retrieval window
- Weak embeddings
- Overly restrictive filtering
- Poor chunk coverage

Typical fixes:
- Increase top-k
- Multi-query retrieval
- Parent-child retrieval
- Better indexing
- Query expansion
- Improve document ingestion

Output Format:

# RAG Evaluation Analysis

## Metric Summary
- Metric: <metric_name>
- Current Score: <score>
- Expected Threshold: <threshold>
- Severity: <Low | Moderate | Severe>

## Metric Interpretation
Explain what this metric represents and what the current score indicates.

## Likely Root Causes
Provide 3 to 7 probable technical causes.

## Recommended Fixes

### High Impact Recommendations
Provide the most important fixes first.

For each recommendation include:
- Problem Addressed
- Recommendation
- Expected Impact
- Tradeoffs / Considerations

### Medium Impact Recommendations

### Advanced Optimizations

## Suggested Pipeline Improvements
Provide end-to-end pipeline-level suggestions if applicable.

## Example Improvements
Provide examples of:
- Better prompts
- Better chunking
- Retrieval improvements
- Re-ranking strategies
- Context formatting

## Final Assessment
Provide a concise concluding assessment including:
- Estimated effort level
- Expected improvement potential
- Priority of remediation

Response Style:
- Professional
- Technical
- Structured
- Concise but detailed
- Easy for engineers and researchers to act upon"""

    user_prompt = f"""Analyze the following RAG system performance data and generate recommendations as per your instructions.

Payload:
{{
    "worst_metric": "{worst_metric}",
    "score": {worst_score},
    "threshold": {threshold}
}}
"""

    if is_ollama:
        if evaluator_model in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]:
            evaluator_model = "gemma4:31b-cloud"
        llm = ChatOllama(model=evaluator_model, temperature=0.0)
    else:
        llm = ChatOpenAI(model=evaluator_model, temperature=0.0)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    return response.content
