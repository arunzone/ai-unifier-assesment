from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.routes.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)


def main():
    settings = get_cached_settings()
    uvicorn.run(
        "ai_unifier_assesment.app:app",
        host=settings.fastapi.host,
        port=settings.fastapi.port,
        reload=False,
    )
