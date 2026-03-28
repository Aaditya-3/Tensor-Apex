---
title: Business Policy Compliance Environment
emoji: "✅"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7861
short_description: Policy-aware customer resolution environment for OpenEnv agents.
---

# Business Policy Compliance and Customer Resolution Environment

This repository is an OpenEnv-style evaluation environment for policy-aware support agents.  
Agents must route, prioritize, escalate, request clarification, flag fraud, and respond while following explicit business rules.

## What Makes This Environment Useful

- Multi-domain tickets across billing, technical support, returns, legal, customer success, and spam.
- 43 scenario templates (`10 easy`, `13 medium`, `20 hard`) with reset-time parameter perturbations to reduce memorization.
- Session-isolated API for concurrent judges and agents using `X-Session-Id`.
- Multi-turn state machine (`episode_phase`) for clarification-sensitive workflows.
- Policy versioning (`v1` and `v2`) to prevent fixed-policy memorization.
- Dynamic policy drift in selected scenarios (`v1 -> v2` mid-episode).
- Operational action-cost budgets that reward efficient trajectories.
- Adversarial/deceptive users: contradiction, policy-gaming, keyword traps, sarcasm, mixed-intent threads.
- Partial observability via hidden internal risk flags not exposed in normal observations.
- Multi-agent interaction through `consult_specialist` and specialist feedback loops.
- Adaptive difficulty: when `task_name` is omitted, scenario difficulty responds to recent performance.
- Browser-based Gradio judge UI.

## Policy Sets

Observations include `policy_version` and the active rule list.

`v1`:
- Refunds over `$500` require escalation.
- VIP customers require `high` or `urgent` priority.
- Issues open over `72` hours require `urgent` priority.
- Complaints mentioning legal action or lawsuits require immediate escalation.
- Suspended accounts must be routed to the billing team.

`v2` (all `v1` rules plus):
- Premier accounts require same-day response regardless of issue type.
- Fraud indicators require `flag_fraud` before any resolution action.

## Observation Space

Each observation includes:
- Current email and visible thread.
- Sender tier, account flags, refund amount, and issue age.
- Action history, policy rules, and task objective.
- `policy_version` and `episode_phase`.
- Optional pending policy shift metadata: `policy_shift_pending`, `policy_shift_at_step`, `policy_shift_to`.
- `specialist_feedback` when a specialist review has been requested.
- Only visible account flags (hidden risk flags stay internal for partial observability).

## Action Space

- `categorize`
- `set_priority`
- `draft_response`
- `escalate`
- `mark_spam`
- `request_info`
- `flag_fraud`
- `snooze`
- `consult_specialist`

`flag_fraud` requires `fraud_reason`.  
`snooze` requires `snooze_hours` and can trigger SLA penalties if it crosses the 72-hour threshold.
`consult_specialist` requires `specialist_question`.

## Reward Table

| Component | Trigger | Value | Notes |
| --- | --- | --- | --- |
| Valid action | Any schema-valid action | +0.05 first use, +0.01 repeated type | Repeated action types get reduced immediate reward |
| Policy penalty | Action contradicts active rule | -0.2 | Immediate |
| Policy history penalty | Any policy violation occurred earlier in episode | -0.08 | Applied on final step for persistent demerit |
| SLA crossed during snooze | Snooze pushes age past 72h | -0.1 | Immediate |
| Fraud missed | No `flag_fraud` in fraud scenario | -0.15 | Episode-end penalty |
| Partial progress | Correct work completed so far | 0.0-current task score | Returned each step |
| Final grader score | Episode complete | 0.0-1.0 | Full grader runs on `done` |
| Efficiency bonus | Finished in <= half allowed steps | +0.1 | Episode-end bonus |
| Cost adjustment | Stay under / exceed scenario budget | +0.08 max / -0.12 max | Episode-end tradeoff term |
| Redundancy penalty | Repeated action types | -0.05 per repeat | Episode-end penalty |
| Running over-budget signal | Cost already above budget before done | down to -0.08 | Intermediate shaping |
| Delayed fraud penalty | Fraud flagged too late in delayed-detection scenarios | up to -0.12 | Episode-end proportional penalty by lateness |
| Early misroute penalty | First categorization mismatches expected route | -0.08 | Episode-end penalty |

Shaped rewards are clamped to `[-1.0, 1.0]` (grader score remains `0.0-1.0`).  
Step details are available in `info["reward_breakdown"]`.
Cost units are normalized operational effort units (roughly comparable to a combined latency/compute/support-minute budget).

## Evaluation Metrics

Each step returns interpretable metrics in `info["evaluation_metrics"]`:

```json
{
  "score": 0.78,
  "policy_score": 0.90,
  "efficiency": 0.60,
  "latency": 0.42,
  "customer_quality": 0.71,
  "risk_management": 0.83,
  "adversarial_resilience": 0.66
}
```

Failure reasons are exposed in `info["failure_modes"]` (for example: `adversarial_miss`, `risk_handling_gap`, `high_operational_cost`).

## API

FastAPI routes:
- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `DELETE /session`

Session header:
- `X-Session-Id` (optional, defaults to `default`)

Important behavior:
- `/step` requires an existing session (call `/reset` first for that session).
- `/state` is session-scoped.
- `/state` hides ground-truth by default; use `/state?include_ground_truth=true` for evaluator/debug mode.

## Gradio Judge UI

`gradio_app.py` starts:
- FastAPI server on `7860`
- Gradio UI on `7861`

The UI supports scenario selection, reset, all action types, and live reward/component readouts.
It also includes a failure-modes dashboard with live evaluation metric breakdowns.

## Verified Rule Baseline Scores

From:
```bash
.venv/bin/python baseline.py --agent rule --seed 42
```

Current means (43 scenarios, latest local run):
- Easy: `0.91` (`std 0.1375`, min `0.70`, max `1.00`)
- Medium: `0.6946` (`std 0.1340`, min `0.4697`, max `0.8397`)
- Hard: `0.5266` (`std 0.1328`, min `0.2751`, max `0.7490`)

Baseline runs are deterministic by default (fixed variation seed). Override with `--seed` when you want alternate perturbation sweeps.

## Project Layout

```text
business_policy_env/
    __init__.py
    baseline.py
    data_generation.py
    environment.py
    models.py
    policies.py
    rewards.py
    server.py
    tasks.py
gradio_app.py
baseline.py
openenv.yaml
pyproject.toml
tests/
    test_environment.py
scripts/
    self_check.sh
    validate_openenv_contract.py
    docker_smoke.sh
.github/workflows/ci.yml
Dockerfile
```

## Local Run

Install:
```bash
.venv/bin/pip install -e ".[dev]"
```

Run API only:
```bash
.venv/bin/uvicorn business_policy_env.server:app --host 0.0.0.0 --port 7860
```

Run Gradio + API:
```bash
.venv/bin/python gradio_app.py
```

Run checks:
```bash
.venv/bin/ruff check .
.venv/bin/mypy business_policy_env/
.venv/bin/python -m pytest tests/ -v
```

One-command self-check (includes OpenEnv contract validation, plus optional `openenv` CLI and Docker smoke checks when installed):
```bash
bash scripts/self_check.sh
```

Contract validation and runtime-proof helper scripts:
- `scripts/validate_openenv_contract.py`
- `scripts/docker_smoke.sh`

CI includes a dedicated `runtime-proof` job that uploads validation/runtime artifacts.

## Docker

The Docker image:
- Installs with `pip install -e .`
- Exposes ports `7860` (API) and `7861` (Gradio)
- Health checks `http://127.0.0.1:7860/health`
- Starts `python gradio_app.py`
