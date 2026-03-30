#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from business_policy_env.baseline import run_baseline


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _configure_openai_env(model_override: str | None) -> str:
    api_base_url = _require_env("API_BASE_URL")
    hf_token = _require_env("HF_TOKEN")
    model_name = model_override or _require_env("MODEL_NAME")

    # Normalize variable names expected by the OpenAI client path.
    os.environ.setdefault("OPENAI_BASE_URL", api_base_url)
    os.environ.setdefault("OPENAI_API_KEY", hf_token)
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenEnv baseline inference script. Default mode runs OpenAI-backed baseline across easy/medium/hard."
    )
    parser.add_argument("--agent", choices=["openai", "rule"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        if args.agent == "openai":
            model_name = _configure_openai_env(args.model)
        else:
            model_name = args.model or os.getenv("MODEL_NAME", "gpt-4.1-mini")

        summary = run_baseline(agent_name=args.agent, model=model_name, seed=args.seed)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"inference.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
