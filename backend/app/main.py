from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.emergency import router as emergency_router
from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Loads LangChain orchestration agents at startup (architecture diagram alignment).
    import app.orchestration.chains  # noqa: F401

    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_v1_prefix)
app.include_router(emergency_router, prefix=settings.api_v1_prefix)
