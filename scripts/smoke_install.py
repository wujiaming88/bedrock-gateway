#!/usr/bin/env python3
"""Verify a clean wheel install, dependencies, and startup imports."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bedrock-gateway-install-") as tmp:
        workspace = Path(tmp)
        venv = workspace / "venv"
        wheels = workspace / "wheels"
        neutral = workspace / "neutral"
        wheels.mkdir()
        neutral.mkdir()

        run(sys.executable, "-m", "venv", str(venv))
        python = venv / "bin" / "python"
        run(
            str(python), "-m", "pip", "wheel", "--no-deps",
            "--wheel-dir", str(wheels), str(ROOT),
        )
        wheel = next(wheels.glob("bedrock_gateway-*.whl"))
        run(str(python), "-m", "pip", "install", str(wheel))
        run(str(python), "-m", "pip", "check")
        for command in ("bedrock-gateway", "muxlane"):
            script = venv / "bin" / command
            if not script.exists():
                raise AssertionError(f"missing CLI entry point: {command}")

        code = """
import importlib.metadata
import bedrock_gateway
import bedrock_gateway.server
import fastapi
import httpx
import multipart
import uvicorn
import yaml

installed = importlib.metadata.version("bedrock-gateway")
assert installed == bedrock_gateway.__version__, (installed, bedrock_gateway.__version__)
print(f"clean install OK: {installed} ({bedrock_gateway.__file__})")
"""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run([str(python), "-c", code], cwd=neutral, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
