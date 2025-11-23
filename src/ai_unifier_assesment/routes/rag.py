from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai_unifier_assesment.rag.qa_service import QAService

router = APIRouter(prefix="/rag", tags=["RAG"])


class QuestionRequest(BaseModel):
    question: str
    collection_name: str = "rag_corpus"


class SourceInfo(BaseModel):
    source: str
    page: int | str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    retrieval_time_ms: float


class DocumentInfo(BaseModel):
    content: str
    source: str
    page: int | str


class RetrieveResponse(BaseModel):
    documents: list[DocumentInfo]
    retrieval_time_ms: float


@router.post("/qa", response_model=AnswerResponse)
async def question_answer(
    request: QuestionRequest,
    qa_service: Annotated[QAService, Depends(QAService)],
) -> AnswerResponse:
    result = qa_service.answer(request.question, request.collection_name)
    return AnswerResponse(**result)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(
    request: QuestionRequest,
    qa_service: Annotated[QAService, Depends(QAService)],
) -> RetrieveResponse:
    result = qa_service.retrieve_only(request.question, request.collection_name)
    return RetrieveResponse(**result)
