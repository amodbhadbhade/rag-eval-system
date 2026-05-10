from pymongo import MongoClient
import os
import certifi

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"), tlsCAFile=certifi.where()
)
db = client["rag_eval"]


def save_dataset(dataset_id, data, name=None):
    """Save a dataset with optional name"""
    db.datasets.insert_one(
        {"dataset_id": dataset_id, "name": name or dataset_id, "data": data}
    )


def get_dataset(dataset_id):
    """Get a dataset by ID"""
    return db.datasets.find_one({"dataset_id": dataset_id})


def list_all_datasets():
    """List all datasets with their IDs and names"""
    datasets = list(db.datasets.find({}, {"dataset_id": 1, "name": 1}))
    return [
        {"dataset_id": d["dataset_id"], "name": d.get("name", d["dataset_id"])}
        for d in datasets
    ]


def get_dataset_questions(dataset_id):
    """Get questions and ground truth answers for a dataset"""
    dataset = db.datasets.find_one({"dataset_id": dataset_id})
    if not dataset:
        return None

    data = dataset["data"]
    questions = data.get("question", [])
    ground_truth = data.get("ground_truth", [])

    return {
        "dataset_id": dataset_id,
        "name": dataset.get("name", dataset_id),
        "questions_and_answers": [
            {"question": q, "ground_truth": gt}
            for q, gt in zip(questions, ground_truth)
        ],
    }


def save_evaluation(payload: dict, scores: dict, framework: str):
    """Save the evaluation payload and scores"""
    record = {**payload, "scores": scores, "framework": framework}
    db.evaluations.insert_one(record)


def get_evaluation_history():
    """Retrieve all evaluation records, excluding the MongoDB _id field."""
    return list(db.evaluations.find({}, {"_id": 0}))
