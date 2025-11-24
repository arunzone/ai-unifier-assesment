from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.routes.chat import router as chat_router
from ai_unifier_assesment.routes.rag import router as rag_router
from ai_unifier_assesment.routes.agent import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(agent_router)


def main():
    settings = get_cached_settings()
    uvicorn.run(
        "ai_unifier_assesment.app:app",
        host=settings.fastapi.host,
        port=settings.fastapi.port,
        reload=False,
    )
