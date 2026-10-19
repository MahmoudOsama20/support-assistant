"""Run the service: python src/serve.py [--host 127.0.0.1] [--port 8000]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import uvicorn  # noqa: E402

from service.app import app  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--demo", action="store_true", help="Run with lightweight demo agent for memory-constrained systems")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    server_app = app
    if args.demo or os.getenv("DEMO_MODE") == "1":
        from agent.demo_agent import DemoAgent
        from service.app import create_app
        logging.info("Starting Nile Wallet service in DEMO mode (lightweight agent)")
        server_app = create_app(agent=DemoAgent())

    uvicorn.run(server_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()