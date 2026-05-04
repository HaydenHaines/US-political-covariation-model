"""Governor forecast router package."""
from fastapi import APIRouter

from api.routers.governor.overview import router as _overview_router
from api.routers.governor.simulation import router as _simulation_router

router = APIRouter()
router.include_router(_overview_router)
router.include_router(_simulation_router)
