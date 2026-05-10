from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from app.services.rag_pipeline import generate_dataset
from app.db.mongo import list_all_datasets, get_dataset_questions
from app.models import (
    CreateDatasetResponse,
    DatasetListResponse,
    DatasetQuestionsResponse,
)

router = APIRouter()


@router.post(
    "/generate",
    response_model=CreateDatasetResponse,
    summary="Generate a new dataset from a PDF file",
    description="Upload a PDF file and generate a question-answer dataset from it",
)
async def create_dataset(file: UploadFile = File(...), num_questions: int = Form(10)):
    """
    Generate a new dataset from a PDF file.

    - **file**: PDF file to process
    - **num_questions**: Number of questions to generate (default: 10)

    Returns the newly created dataset_id
    """
    dataset_id = await generate_dataset(file, num_questions)
    return CreateDatasetResponse(
        dataset_id=dataset_id,
        message=f"Dataset created successfully with {num_questions} questions",
    )


@router.get(
    "/list",
    response_model=DatasetListResponse,
    summary="List all datasets",
    description="Retrieve a list of all available datasets with their IDs and names",
)
def list_datasets():
    """
    List all available datasets.

    Returns a list of datasets with their IDs and names.
    """
    datasets = list_all_datasets()
    return DatasetListResponse(datasets=datasets, total_count=len(datasets))


@router.get(
    "/{dataset_id}/questions",
    response_model=DatasetQuestionsResponse,
    summary="Get questions and answers for a dataset",
    description="Retrieve all gold questions and ground truth answers for a specific dataset",
)
def get_dataset_qna(dataset_id: str):
    """
    Get questions and ground truth answers for a dataset.

    - **dataset_id**: The ID of the dataset to retrieve

    Returns the dataset with all questions and their corresponding ground truth answers.
    """
    result = get_dataset_questions(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    return DatasetQuestionsResponse(
        dataset_id=result["dataset_id"],
        name=result["name"],
        questions_and_answers=result["questions_and_answers"],
        total_questions=len(result["questions_and_answers"]),
    )
