#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from business_policy_env.server import app

REQUIRED_ENDPOINTS = {"health", "tasks", "reset", "step", "state", "close_session"}
REQUIRED_TASKS = {"easy", "medium", "hard"}
REQUIRED_ACTIONS = {
    "categorize",
    "set_priority",
    "draft_response",
    "escalate",
    "mark_spam",
    "request_info",
    "flag_fraud",
    "snooze",
    "consult_specialist",
}


def _parse_openenv(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    endpoints: dict[str, str] = {}
    actions: list[str] = []
    tasks: list[dict[str, str]] = []

    current_section: str | None = None
    for line in lines:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", line):
            current_section = line.split(":", 1)[0].strip()
            continue

        if current_section == "endpoints":
            match = re.match(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\S+)\s*$", line)
            if match:
                endpoints[match.group(1)] = match.group(2)
            continue

        if current_section == "actions":
            match = re.match(r"^\s{2}-\s*(\S+)\s*$", line)
            if match:
                actions.append(match.group(1))
            continue

        if current_section == "tasks":
            match = re.match(r"^\s{2}-\s*name:\s*(\S+)\s*$", line)
            if match:
                tasks.append({"name": match.group(1)})

    return {"endpoints": endpoints, "actions": actions, "tasks": tasks}


def _routes() -> set[str]:
    return {route.path for route in app.routes}


def main() -> None:
    config_path = Path("openenv.yaml")
    data = _parse_openenv(config_path)
    errors: list[str] = []

    endpoints = data.get("endpoints", {})
    if not isinstance(endpoints, dict):
        errors.append("endpoints must be a mapping.")
        endpoints = {}

    endpoint_keys = set(endpoints.keys())
    missing_endpoint_keys = sorted(REQUIRED_ENDPOINTS - endpoint_keys)
    if missing_endpoint_keys:
        errors.append(f"Missing endpoint keys: {missing_endpoint_keys}")

    app_routes = _routes()
    configured_paths = {str(path) for path in endpoints.values()}
    missing_paths = sorted(path for path in configured_paths if path not in app_routes)
    if missing_paths:
        errors.append(f"Configured endpoint paths not found in FastAPI app: {missing_paths}")

    actions = data.get("actions", [])
    if not isinstance(actions, list):
        errors.append("actions must be a list.")
        actions = []
    action_set = {str(action) for action in actions}
    missing_actions = sorted(REQUIRED_ACTIONS - action_set)
    if missing_actions:
        errors.append(f"Missing required actions: {missing_actions}")

    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        errors.append("tasks must be a list.")
        tasks = []
    task_names = {str(task.get("name")) for task in tasks if isinstance(task, dict)}
    missing_tasks = sorted(REQUIRED_TASKS - task_names)
    if missing_tasks:
        errors.append(f"Missing required task names: {missing_tasks}")

    report = {
        "ok": not errors,
        "endpoint_keys": sorted(endpoint_keys),
        "configured_paths": sorted(configured_paths),
        "app_routes": sorted(app_routes),
        "actions": sorted(action_set),
        "tasks": sorted(task_names),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
