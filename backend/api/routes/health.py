"""
Health Check Routes

Endpoints for monitoring application health.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from core.config import get_settings
from services.teaching_service import get_teaching_service

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    service: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check with service stats."""
    status: str
    version: str
    service: str
    stats: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        service=settings.app_name
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with service statistics.
    
    Returns:
        Detailed health status with stats
    """
    settings = get_settings()
    
    try:
        teaching_service = get_teaching_service()
        stats = teaching_service.get_stats()
        status = "healthy"
    except Exception as e:
        stats = {"error": str(e)}
        status = "degraded"
    
    return DetailedHealthResponse(
        status=status,
        version=settings.app_version,
        service=settings.app_name,
        stats=stats
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
