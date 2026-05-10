import os
from dotenv import load_dotenv

load_dotenv()
import uuid
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.db.mongo import save_dataset
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_ollama import ChatOllama, OllamaEmbeddings


# class NativeDoclingLoader:
#     """Custom LangChain-compatible loader using native Docling."""

#     def __init__(self, file_path: str):
#         self.file_path = file_path

#     def load(self):
#         pipeline_options = PdfPipelineOptions()
#         pipeline_options.do_ocr = (
#             False  # Disable OCR to avoid loading heavy Tesseract/EasyOCR models
#         )
#         pipeline_options.do_table_structure = (
#             True  # Keep structure; disable if you only need raw text
#         )
#         pipeline_options.generate_page_images = (
#             False  # Critical: prevents storing large bitmaps in memory
#         )
#         converter = DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
#                 )
#             }
#         )
#         # converter = DocumentConverter()
#         result = converter.convert(self.file_path)
#         text = result.document.export_to_markdown()
#         return [Document(page_content=text, metadata={"source": self.file_path})]


def classify_pdf(file):
    """
    Analyzes PDF to determine the best loader:
    - 'scanned': For Unstructured (OCR needed)
    - 'complex': For Docling (Tables/Layouts)
    - 'simple': For PyMuPDF (Fast text)
    """
    doc = fitz.open(file)
    total_pages = len(doc)
    text_pages = 0
    table_pages = 0

    # Check up to first 5 pages to save time
    check_limit = min(total_pages, 5)

    for i in range(check_limit):
        page = doc.load_page(i)
        text = page.get_text().strip()

        # 1. Detection: Scanned vs. Digital
        # If page has < 50 chars, it's likely an image/scanned
        if len(text) > 50:
            text_pages += 1

        # 2. Detection: Tables/Complex Layouts
        # PyMuPDF's built-in table finder is fast
        tables = page.find_tables()
        if tables.tables:
            table_pages += 1

    doc.close()

    # DECISION LOGIC
    # If >50% of checked pages have no text, treat as Scanned
    if text_pages / check_limit < 0.5:
        return "scanned"

    # If tables are found in valid text pages, treat as Complex
    elif table_pages > 0:
        return "complex"

    # Otherwise, it's standard linear text
    return "simple"


def get_loader(file_path):
    # category = classify_pdf(file_path)

    # # if category == "scanned":
    # #     print("Routing to Unstructured (OCR)...")
    # #     return UnstructuredPDFLoader(file_path, mode="elements", strategy="ocr_only")

    # if category == "complex":
    #     print("Routing to Docling (Structure/Table preserve)...")
    #     return NativeDoclingLoader(file_path)

    # else:  # simple
    #     print("Routing to PyMuPDF (Fast)...")
    #     return PyMuPDFLoader(file_path)
    return PyMuPDFLoader(file_path)


async def generate_dataset(file, num_questions):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # loader = PyPDFLoader(file_path)
    loader = get_loader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    # Free Cloud-based Generator
    generator_model = ChatOllama(
        model=os.getenv("GENERATOR_LLM", "gemma4:31b-cloud"),
        temperature=os.getenv("GENERATOR_LLM_TEMPERATURE", 0.0),
    )

    # Free Cloud-based Critic (Flash version for speed/free tier)
    critic_model = ChatOllama(
        model=os.getenv("CRITIC_LLM", "gemma4:31b-cloud"),
        temperature=os.getenv("CRITIC_LLM_TEMPERATURE", 0.0),
    )

    # 2. Local Embedding for speed/privacy
    # nomic handles the long context (8k tokens) required for Ragas chunks
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )

    # Initialize generator
    # generator = TestsetGenerator(llm=generator_llm, critic_llm=critic_llm)

    # Configure Ragas testset generator with OpenAI models
    # generator_llm = ChatOpenAI(model="gpt-4o-mini")
    # critic_llm = ChatOpenAI(model="gpt-4o-mini")
    # embeddings = OpenAIEmbeddings()

    # Push this as high as your LLM provider's Tier/Rate Limits allow
    # For free cloud tiers or local Ollama, lower max_workers prevents 429 errors.
    my_run_config = RunConfig(
        max_workers=2,
        timeout=120,  # Increased timeout for slower models
        max_retries=5,  # Increased retries to handle transient rate limits
    )

    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_model,
        critic_llm=critic_model,
        embeddings=embeddings,
    )
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=num_questions,
        distributions={simple: 1},
        run_config=my_run_config,
    )

    dataset_id = str(uuid.uuid4())
    # Extract dataset name from filename (remove extension)
    dataset_name = file.filename.rsplit(".", 1)[0] if file.filename else dataset_id
    save_dataset(
        dataset_id, testset.to_pandas().to_dict(orient="list"), name=dataset_name
    )

    return dataset_id
