"""Loopback HTTP adapter; disconnect cancellation does not interrupt native work."""

import asyncio
import math
import re
import resource
import sys

from aiohttp import web
import aiohttp
import platform
import psutil

from .backend import ProcessBackend
from .scheduler import Scheduler

STATE = web.AppKey("scheduler", Scheduler)
BACKEND = web.AppKey("backend", ProcessBackend)
ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


async def infer(request):
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {"id", "input", "timeout_ms"}:
            raise ValueError()
        identifier, values, timeout = body["id"], body["input"], body["timeout_ms"]
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError()
        if (not isinstance(values, list) or len(values) != 64 or
                any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in values)):
            raise ValueError()
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60000:
            raise ValueError()
    except (ValueError, TypeError, KeyError, OverflowError):
        return web.json_response({"status": "invalid_request"}, status=400)
    scheduler = request.app[STATE]
    try:
        future = scheduler.submit(identifier, values, timeout)
    except ValueError:
        return web.json_response({"id": identifier, "status": "duplicate_id"}, status=409)
    try:
        result = await asyncio.shield(future)
    except asyncio.CancelledError:
        scheduler.cancel(identifier, reason="client_disconnect")
        raise
    return web.json_response(result, status=result["http_status"])


async def cancel(request):
    scheduler = request.app[STATE]
    identifier = request.match_info["identifier"]
    if identifier not in scheduler.records and identifier not in scheduler.entries:
        return web.json_response({"id": identifier, "cancelled": False}, status=404)
    changed = scheduler.cancel(identifier)
    return web.json_response({"id": identifier, "cancelled": changed}, status=200 if changed else 409)


async def health(request):
    scheduler = request.app[STATE]
    return web.json_response({"ready": scheduler.healthy and not scheduler.closing,
                              "worker": request.app[BACKEND].metadata,
                              "http_environment": {"python": platform.python_version(), "aiohttp": aiohttp.__version__, "psutil": psutil.__version__}},
                             status=200 if scheduler.healthy and not scheduler.closing else 503)


async def stats(request):
    result = request.app[STATE].snapshot()
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["http_high_water_rss_bytes"] = int(value if sys.platform == "darwin" else value * 1024)
    result["worker_high_water_rss_bytes"] = request.app[BACKEND].highwater
    return web.json_response(result)


def create_app(model, *, batch_size=16, wait_ms=2., capacity=128, audit_limit=4096, backend_factory=ProcessBackend):
    app = web.Application(client_max_size=8192)
    async def lifecycle(app):
        backend = backend_factory(model)
        app[BACKEND] = backend
        try:
            await backend.ready()
            scheduler = Scheduler(backend, batch_size=batch_size, wait_ms=wait_ms,
                                  capacity=capacity, audit_limit=audit_limit)
            app[STATE] = scheduler
            scheduler.start()
            try:
                yield
            finally:
                await scheduler.close()
        finally:
            await backend.close()
    app.cleanup_ctx.append(lifecycle)
    app.router.add_post("/v1/infer", infer)
    app.router.add_delete("/v1/requests/{identifier}", cancel)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/stats", stats)
    return app
