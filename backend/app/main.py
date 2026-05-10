from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import dataset, evaluate

# Create FastAPI app with OpenAPI/Swagger documentation
app = FastAPI(
    title="RAG Evaluation System API",
    description="API for managing datasets and evaluating RAG (Retrieval-Augmented Generation) responses using RAGAS metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with tags for Swagger organization
app.include_router(
    dataset.router,
    prefix="/dataset",
    tags=["Dataset Management"],
)
app.include_router(
    evaluate.router,
    prefix="/evaluate",
    tags=["Evaluation"],
)
