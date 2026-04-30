"""
Dashboard package for Bedrock Gateway.

Exposes:
  - ``MetricsCollector`` — thread-safe in-memory metrics store
  - ``build_dashboard_router`` — FastAPI router serving UI + metrics API
  - ``metrics_middleware_factory`` — request-timing middleware
  - ``DashboardAuth`` — API-key/localhost gate for the dashboard
  - ``RateLimiter`` — per-IP fixed-window limiter for the metrics API
"""

from __future__ import annotations

from .api import build_dashboard_router
from .metrics import MetricsCollector
from .middleware import metrics_middleware_factory
from .security import DashboardAuth, RateLimiter
from .storage import MetricsStorage

__all__ = [
    "DashboardAuth",
    "MetricsCollector",
    "MetricsStorage",
    "RateLimiter",
    "build_dashboard_router",
    "metrics_middleware_factory",
]
