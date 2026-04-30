"""
FastAPI router wiring the dashboard UI and metrics endpoints.

Endpoints:

    GET /dashboard/                 — HTML shell (served from static/)
    GET /dashboard/login            — simple API-key login form
    POST /dashboard/login           — validates key, sets cookie, redirects
    GET /dashboard/logout           — clears the cookie
    GET /api/metrics/overview       — top-of-page summary tiles
    GET /api/metrics/traffic        — per-minute QPS + latency series
    GET /api/metrics/models         — per-model usage (requests + tokens)
    GET /api/metrics/requests       — recent request log (filter/limit)
    GET /api/metrics/errors         — error counts + recent errors
    GET /api/metrics/system         — version / uptime / region / auth mode

All dashboard + metrics responses carry a set of hardening headers
(CSP, X-Frame-Options, etc.) and every ``/api/metrics/*`` hit is
subject to a per-IP rate limit.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from .metrics import MetricsCollector
from .security import (
    SECURITY_HEADERS,
    DashboardAuth,
    RateLimiter,
    mask_ip,
    sanitize_request_log,
)


_STATIC_DIR = Path(__file__).parent / "static"


# Window label → minutes of history
_WINDOW_MINUTES: dict[str, int] = {
    "1h": 60,
    "6h": 6 * 60,
    "24h": 24 * 60,
}


_COOKIE_NAME = "bedrock_gw_key"


def _system_info(request: Request) -> dict[str, Any]:
    """Pull runtime bits from ``app.state`` without tightly coupling modules."""
    from .. import __version__

    app = request.app
    state = app.state

    config = getattr(state, "config", None)
    registry = getattr(state, "registry", None)
    auth = getattr(state, "auth", None)

    region = getattr(config, "region", "-") if config else "-"
    auth_mode = getattr(auth, "mode", "-") if auth else "-"
    try:
        model_count = len(registry.list_models()) if registry else 0
    except Exception:
        model_count = 0

    return {
        "version": __version__,
        "auth_mode": auth_mode,
        "region": region,
        "model_count": model_count,
    }


def _apply_security_headers(response: Response) -> Response:
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


def _unauthorized_json() -> JSONResponse:
    resp = JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Dashboard authentication required",
                "type": "authentication_error",
                "code": 401,
            }
        },
    )
    return _apply_security_headers(resp)


def _forbidden_json(reason: str) -> JSONResponse:
    resp = JSONResponse(
        status_code=403,
        content={
            "error": {
                "message": f"Dashboard access denied: {reason}",
                "type": "permission_error",
                "code": 403,
            }
        },
    )
    return _apply_security_headers(resp)


def _login_page(error: str | None = None, next_url: str = "/dashboard/") -> HTMLResponse:
    err_html = (
        f'<p class="err">{html.escape(error)}</p>' if error else ""
    )
    safe_next = html.escape(next_url)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Bedrock Gateway — Sign in</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0b0f14;color:#d6dde6;
     display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}}
form{{background:#121821;padding:2rem;border-radius:8px;min-width:320px;
     box-shadow:0 10px 30px rgba(0,0,0,.5);}}
h1{{margin:0 0 1rem 0;font-size:1.1rem;letter-spacing:.08em;}}
label{{display:block;font-size:.8rem;margin-bottom:.4rem;opacity:.7;}}
input{{width:100%;padding:.6rem .8rem;border-radius:4px;border:1px solid #2a3340;
      background:#0b0f14;color:#d6dde6;box-sizing:border-box;font-family:monospace;}}
button{{margin-top:1rem;width:100%;padding:.7rem;border-radius:4px;border:0;
       background:#3b82f6;color:#fff;cursor:pointer;font-weight:600;}}
.err{{color:#f87171;font-size:.85rem;margin:.5rem 0 0 0;}}
</style></head>
<body>
<form method="POST" action="/dashboard/login">
  <h1>BEDROCK GATEWAY — SIGN IN</h1>
  <label for="key">API Key</label>
  <input id="key" name="key" type="password" autocomplete="off" autofocus />
  <input type="hidden" name="next" value="{safe_next}" />
  <button type="submit">Sign in</button>
  {err_html}
</form></body></html>
"""
    resp = HTMLResponse(content=body, status_code=200)
    return _apply_security_headers(resp)


def _parse_form_urlencoded(body: bytes) -> dict[str, str]:
    """Parse application/x-www-form-urlencoded without pulling python-multipart."""
    from urllib.parse import parse_qs

    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return {}
    parsed = parse_qs(text, keep_blank_values=True)
    return {k: (v[0] if v else "") for k, v in parsed.items()}


def build_dashboard_router(
    collector: MetricsCollector,
    *,
    auth: DashboardAuth | None = None,
    rate_limiter: RateLimiter | None = None,
    extra_system_info: Callable[[Request], dict[str, Any]] | None = None,
) -> APIRouter:
    """
    Build a router exposing the metrics API and the static dashboard UI.

    Parameters
    ----------
    collector:
        The shared :class:`MetricsCollector`.
    auth:
        Dashboard access gate. When omitted, an open gate is used
        (no auth, no localhost-only check) — useful for tests.
    rate_limiter:
        Per-IP rate limiter applied to ``/api/metrics/*``. When omitted,
        no limit is enforced.
    extra_system_info:
        Optional override for the ``/api/metrics/system`` payload.
    """
    router = APIRouter()

    if auth is None:
        auth = DashboardAuth(
            enabled=True, api_key="", require_auth=False, localhost_only=False
        )

    def _client_key(request: Request) -> str:
        client = request.client
        if client and client.host:
            return client.host
        return request.headers.get("x-forwarded-for", "-").split(",")[0].strip()

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    def _guard_api(request: Request) -> Response | None:
        """Enforce auth + rate limit for JSON endpoints. Returns a response
        when blocked, or ``None`` when the request should proceed."""
        allowed, reason = auth.check(request)
        if not allowed:
            if reason == "auth_required":
                return _unauthorized_json()
            return _forbidden_json(reason)
        if rate_limiter is not None:
            ok, retry_after = rate_limiter.check(_client_key(request))
            if not ok:
                resp = JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Rate limit exceeded",
                            "type": "rate_limit_error",
                            "code": 429,
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )
                return _apply_security_headers(resp)
        return None

    def _guard_ui(request: Request) -> Response | None:
        """Enforce auth for dashboard UI, redirecting to /dashboard/login."""
        allowed, reason = auth.check(request)
        if allowed:
            return None
        if reason == "auth_required":
            # Redirect browsers to the login page; include ?next=.
            next_url = request.url.path
            if request.url.query:
                next_url += f"?{request.url.query}"
            resp = RedirectResponse(
                url=f"/dashboard/login?next={next_url}", status_code=302
            )
            return _apply_security_headers(resp)
        return _forbidden_json(reason)

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------

    @router.get("/api/metrics/overview")
    async def overview(request: Request) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        return _apply_security_headers(JSONResponse(content=collector.overview()))

    @router.get("/api/metrics/traffic")
    async def traffic(
        request: Request,
        window: str = Query("1h", pattern="^(1h|6h|24h)$"),
    ) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        minutes = _WINDOW_MINUTES.get(window, 60)
        data = collector.timeseries(minutes=minutes)
        data["window"] = window
        return _apply_security_headers(JSONResponse(content=data))

    @router.get("/api/metrics/models")
    async def models(request: Request) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        return _apply_security_headers(JSONResponse(content=collector.model_stats()))

    @router.get("/api/metrics/requests")
    async def recent(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        filter: str = Query("all", pattern="^(all|success|error)$"),
    ) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        items = collector.recent_requests(limit=200)
        if filter == "success":
            items = [r for r in items if r["status"] < 400]
        elif filter == "error":
            items = [r for r in items if r["status"] >= 400]
        items = sanitize_request_log(items[:limit])
        return _apply_security_headers(
            JSONResponse(
                content={"requests": items, "filter": filter, "limit": limit}
            )
        )

    @router.get("/api/metrics/errors")
    async def errors(request: Request) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        breakdown = collector.error_breakdown()
        breakdown["recent"] = sanitize_request_log(
            collector.recent_errors(limit=20)
        )
        return _apply_security_headers(JSONResponse(content=breakdown))

    @router.get("/api/metrics/sources")
    async def sources(
        request: Request,
        limit: int = Query(10, ge=1, le=50),
    ) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        data = collector.sources_stats(top_n=limit)
        # Mask IPs before returning — dashboards don't need the last octet.
        data["sources"] = [
            {**row, "ip": mask_ip(row["ip"])} for row in data["sources"]
        ]
        return _apply_security_headers(JSONResponse(content=data))

    @router.get("/api/metrics/memory")
    async def memory(
        request: Request,
        window: str = Query("1h", pattern="^(1h|6h|24h)$"),
    ) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        minutes = _WINDOW_MINUTES.get(window, 60)
        data = collector.memory_timeseries(minutes=minutes)
        data["window"] = window
        return _apply_security_headers(JSONResponse(content=data))

    @router.get("/api/metrics/system")
    async def system(request: Request) -> Response:
        blocked = _guard_api(request)
        if blocked is not None:
            return blocked
        info = (extra_system_info or _system_info)(request)
        payload = collector.system_status(
            version=info.get("version", "0.0.0"),
            auth_mode=info.get("auth_mode", "-"),
            region=info.get("region", "-"),
            model_count=int(info.get("model_count", 0)),
        )
        return _apply_security_headers(JSONResponse(content=payload))

    # ------------------------------------------------------------------
    # Login / logout
    # ------------------------------------------------------------------

    @router.get("/dashboard/login", include_in_schema=False)
    async def login_form(
        request: Request, next: str = Query("/dashboard/")
    ) -> Response:
        # If no API key configured, send users straight to the dashboard —
        # the localhost-only gate is handled by _guard_ui.
        if not auth.is_configured_key():
            return _apply_security_headers(
                RedirectResponse(url=next or "/dashboard/", status_code=302)
            )
        # Already authenticated?
        allowed, _ = auth.check(request)
        if allowed:
            return _apply_security_headers(
                RedirectResponse(url=next or "/dashboard/", status_code=302)
            )
        return _login_page(next_url=next)

    @router.post("/dashboard/login", include_in_schema=False)
    async def login_submit(request: Request) -> Response:
        body = await request.body()
        ctype = request.headers.get("content-type", "")
        key = ""
        next_url = "/dashboard/"
        if "application/x-www-form-urlencoded" in ctype:
            form = _parse_form_urlencoded(body)
            key = form.get("key", "")
            next_url = form.get("next", "/dashboard/") or "/dashboard/"
        elif "application/json" in ctype:
            try:
                import json as _json

                payload = _json.loads(body or b"{}")
            except Exception:
                payload = {}
            key = str(payload.get("key", ""))
            next_url = str(payload.get("next", "/dashboard/")) or "/dashboard/"
        # Only allow same-site relative redirects.
        if not next_url.startswith("/") or next_url.startswith("//"):
            next_url = "/dashboard/"

        if not auth.verify_key(key):
            return _login_page(error="Invalid API key", next_url=next_url)

        resp = RedirectResponse(url=next_url, status_code=302)
        resp.set_cookie(
            key=_COOKIE_NAME,
            value=auth.api_key,
            httponly=True,
            samesite="strict",
            secure=False,  # served over HTTP in dev; operator can add TLS termination
            path="/",
            max_age=12 * 60 * 60,
        )
        return _apply_security_headers(resp)

    @router.get("/dashboard/logout", include_in_schema=False)
    async def logout() -> Response:
        resp = RedirectResponse(url="/dashboard/login", status_code=302)
        resp.delete_cookie(key=_COOKIE_NAME, path="/")
        return _apply_security_headers(resp)

    # ------------------------------------------------------------------
    # Static UI (guarded)
    # ------------------------------------------------------------------

    @router.get("/dashboard", include_in_schema=False)
    async def dashboard_redirect(request: Request) -> Response:
        blocked = _guard_ui(request)
        if blocked is not None:
            return blocked
        return _apply_security_headers(
            RedirectResponse(url="/dashboard/", status_code=307)
        )

    @router.get("/dashboard/", include_in_schema=False)
    async def dashboard_index(request: Request) -> Response:
        blocked = _guard_ui(request)
        if blocked is not None:
            return blocked
        index = _STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="dashboard UI not installed")
        return _apply_security_headers(
            FileResponse(index, media_type="text/html")
        )

    @router.get("/dashboard/{filename:path}", include_in_schema=False)
    async def dashboard_asset(request: Request, filename: str) -> Response:
        # Login / logout paths are handled by their own routes above.
        if filename in {"login", "logout"}:
            raise HTTPException(status_code=404, detail="not found")
        blocked = _guard_ui(request)
        if blocked is not None:
            return blocked
        # Normalise and block path traversal
        target = (_STATIC_DIR / filename).resolve()
        try:
            target.relative_to(_STATIC_DIR.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="not found") from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return _apply_security_headers(FileResponse(target))

    return router
