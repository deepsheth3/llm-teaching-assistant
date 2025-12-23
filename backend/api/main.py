"""
FastAPI Application

Main entry point for the LLM Teaching Assistant API.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import BaseAppException
from api.routes import teach, leetcode, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        🎓 **LLM Teaching Assistant API**
        
        An AI-powered teaching assistant that:
        - Retrieves and explains research papers from arXiv
        - Converts academic content into beginner-friendly lessons
        - Provides LeetCode problems for coding practice
        
        ## Features
        
        - **Semantic Search**: Find relevant papers using natural language
        - **Lesson Generation**: Get step-by-step explanations
        - **Streaming Support**: Real-time lesson generation via SSE
        - **Coding Practice**: Random LeetCode problems
        
        ## Quick Start
        
        ```python
        import requests
        
        # Generate a lesson
        response = requests.post(
            "http://localhost:8000/api/v1/teach",
            json={"query": "attention mechanisms in transformers"}
        )
        lesson = response.json()
        ```
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Exception handler
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        logger.error(f"Application error: {exc.code} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    # Generic exception handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred"
                }
            }
        )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(teach.router, prefix=settings.api_prefix, tags=["Teaching"])
    app.include_router(leetcode.router, prefix=settings.api_prefix, tags=["LeetCode"])
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
