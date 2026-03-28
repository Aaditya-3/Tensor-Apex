from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import Any

import gradio as gr
import uvicorn

from business_policy_env.baseline import RuleBasedAgent
from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action
from business_policy_env.server import app as fastapi_app
from business_policy_env.tasks import scenario_registry


def start_api() -> None:
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="error")


threading.Thread(target=start_api, daemon=True).start()

env = BusinessPolicyComplianceEnv()
current_obs = None


# ── helpers ──────────────────────────────────────────────────────────────────

def get_scenario_choices() -> list[tuple[str, str]]:
    registry = scenario_registry()
    entries = sorted(registry.values(), key=lambda item: (item.difficulty, item.scenario_id))
    return [(f"[{s.difficulty.upper()}] {s.title}", s.scenario_id) for s in entries]


def format_observation(obs) -> str:
    lines = [
        f"**Scenario:** {obs.scenario_id} [{obs.difficulty}]",
        f"**Policy version:** {obs.policy_version}",
        (
            f"**Policy shift:** pending at step {obs.policy_shift_at_step} → {obs.policy_shift_to}"
            if obs.policy_shift_pending
            else "**Policy shift:** none pending"
        ),
        f"**Phase:** {obs.episode_phase}",
        f"**Steps:** {obs.steps_taken}/{obs.max_steps}",
        f"**Issue age:** {obs.issue_age_hours:.1f}h",
        f"**Sender tier:** {obs.sender_tier}",
        f"**Refund amount:** {'$' + str(obs.refund_amount) if obs.refund_amount is not None else 'N/A'}",
        f"**Account flags:** {', '.join(obs.account_flags) if obs.account_flags else 'none'}",
        "",
        "**Latest email:**",
        f"> Subject: {obs.current_email.subject}",
        f"> {obs.current_email.body}",
        "",
        "**Policy rules:**",
    ]
    for rule in obs.policy_rules:
        lines.append(f"- {rule}")
    if obs.action_history:
        lines.append("")
        lines.append("**Action history:**")
        for record in obs.action_history:
            lines.append(f"- Step {record.step_index}: `{record.action.action_type}`")
    return "\n".join(lines)


def _action_label(action: Action) -> str:
    parts = [f"`{action.action_type}`"]
    if action.category:
        parts.append(f"→ **{action.category}**")
    if action.priority:
        parts.append(f"→ **{action.priority}**")
    if action.escalation_reason:
        parts.append(f"— _{action.escalation_reason}_")
    if action.fraud_reason:
        parts.append(f"— _{action.fraud_reason}_")
    if action.clarifying_question:
        parts.append(f"— _{action.clarifying_question}_")
    if action.response_text:
        preview = action.response_text[:80] + ("…" if len(action.response_text) > 80 else "")
        parts.append(f'— "{preview}"')
    if action.snooze_hours:
        parts.append(f"→ {action.snooze_hours}h")
    return " ".join(parts)


# ── Manual tab ────────────────────────────────────────────────────────────────

def reset_episode(scenario_id: str) -> tuple[str, str, str, dict[str, Any]]:
    global current_obs
    current_obs = env.reset(scenario_id=scenario_id)
    return (
        format_observation(current_obs),
        "",
        "Episode reset. Ready for actions.",
        gr.update(interactive=True),
    )


def take_action(
    action_type: str,
    category: str,
    priority: str,
    response_text: str,
    escalation_reason: str,
    clarifying_question: str,
    fraud_reason: str,
    snooze_hours: int,
    specialist_question: str,
    reasoning: str,
) -> tuple[str, str, str]:
    global current_obs
    if current_obs is None:
        return "Reset the environment first.", "", "No active episode."
    try:
        action = Action(
            action_type=action_type,
            reasoning=reasoning or "Manual action from Gradio UI.",
            category=category or None,
            priority=priority or None,
            response_text=response_text or None,
            escalation_reason=escalation_reason or None,
            clarifying_question=clarifying_question or None,
            fraud_reason=fraud_reason or None,
            snooze_hours=snooze_hours or None,
            specialist_question=specialist_question or None,
        )
    except Exception as exc:
        return format_observation(current_obs), "", f"Invalid action: {exc}"

    current_obs, reward, done, info = env.step(action)
    status = f"Reward: {reward:.4f} | Done: {done}"
    if info.get("policy_violations"):
        status += f" | ⚠️ Policy violations: {', '.join(info['policy_violations'])}"
    if info.get("policy_event"):
        status += f" | 🔄 {info['policy_event']}"
    rb = info.get("reward_breakdown", {})
    if "cost_spent" in rb and "cost_budget" in rb:
        status += f" | Cost: {rb['cost_spent']}/{rb['cost_budget']}"
    if done:
        status += f" | 🏁 Final score: {info.get('final_score', 0):.4f}"
    return format_observation(current_obs), str(info.get("component_scores", {})), status


# ── Demo tab ──────────────────────────────────────────────────────────────────

_DEMO_SCENARIOS: list[tuple[str, str]] = [
    ("[HARD]   Hidden fraud with delayed-detection risk",  "hard_hidden_fraud_delayed_detection"),
    ("[MEDIUM] Adversarial duplicate-refund claim",        "medium_adversarial_refund_already_processed"),
    ("[EASY]   VIP refund over $500 threshold",            "easy_vip_refund"),
    ("[EASY]   SLA breach — aged billing ticket",          "easy_sla_breach"),
    ("[EASY]   Legal threat triggers escalation",          "easy_legal_threat"),
    ("[MEDIUM] Ambiguous charge — clarification required", "medium_charge_or_bug"),
    ("[MEDIUM] Policy-gaming disguised as urgency",        "medium_policy_gaming_subtle"),
    ("[HARD]   VIP refund + legal pressure + SLA breach",  "hard_vip_refund_lawyer"),
    ("[HARD]   Chargeback fraud detection",                "hard_fraud_chargeback"),
    ("[HARD]   Sarcastic multilingual leaderboard trap",   "hard_sarcastic_multilingual_trap"),
    ("[HARD]   Long-horizon escalation chain (18 steps)",  "hard_long_horizon_escalation_chain"),
    ("[HARD]   Three conflicting policy signals",          "hard_three_signal_precedence"),
]

_DIFFICULTY_EMOJI = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

_VIOLATION_TIPS: dict[str, str] = {
    "Priority must be at least": "💡 Policy requires higher priority for this sender tier or issue age.",
    "escalation before resolution": "💡 High-value refund or legal threat must be escalated first.",
    "flag_fraud before resolution": "💡 Fraud indicators detected — flag fraud before routing.",
    "Category must be": "💡 Suspended accounts must route to billing.",
}


def _violation_hint(violations: list[str]) -> str:
    for v in violations:
        for key, tip in _VIOLATION_TIPS.items():
            if key in v:
                return tip
    return ""


def run_demo(
    scenario_id: str,
    step_delay: float,
    show_reasoning: bool,
) -> Generator[tuple[str, str], None, None]:
    """Stream demo log lines and a live scoreboard as the rule agent plays."""

    demo_env = BusinessPolicyComplianceEnv()
    agent = RuleBasedAgent()
    obs = demo_env.reset(scenario_id=scenario_id)

    scenario = scenario_registry()[scenario_id]
    diff_emoji = _DIFFICULTY_EMOJI.get(scenario.difficulty, "⚪")
    policy_shift_note = (
        f" → shifting to {obs.policy_shift_to} at step {obs.policy_shift_at_step}" if obs.policy_shift_pending else ""
    )

    log_lines: list[str] = [
        f"## {diff_emoji} {scenario.title}",
        f"> **Objective:** {scenario.objective}",
        f"> **Policy:** {obs.policy_version}{policy_shift_note}",
        f"> **Max steps:** {obs.max_steps} | **Cost budget:** ${scenario.cost_budget:.2f}",
        "",
        f"**📧 Ticket ({obs.sender_tier.upper()} | age {obs.issue_age_hours:.1f}h)**",
        f"> *{obs.current_email.subject}*",
        f"> {obs.current_email.body}",
        "",
        "---",
        "### 🤖 Agent decisions",
        "",
    ]

    scoreboard = ""
    yield "\n".join(log_lines), scoreboard

    done = False
    step = 0
    final_score = 0.0
    component_scores: dict[str, float] = {}
    previous_action_type: str | None = None
    repeated_action_streak = 0
    budget_warning_logged = False

    while not done:
        action = agent.next_action(obs)
        obs, reward, done, info = demo_env.step(action)
        step += 1
        if action.action_type == previous_action_type:
            repeated_action_streak += 1
        else:
            repeated_action_streak = 1
        previous_action_type = action.action_type

        policy_event = info.get("policy_event") or ""
        violations = info.get("policy_violations", [])
        components = info.get("component_scores", {})
        rb = info.get("reward_breakdown", {})
        failure_modes = info.get("failure_modes", [])

        # Build step block
        reward_icon = "✅" if reward >= 0 else "❌"
        step_lines: list[str] = [
            f"**Step {step}** — {_action_label(action)} {reward_icon} `{reward:+.4f}`",
        ]
        if show_reasoning and action.reasoning:
            step_lines.append(f"  *Reasoning: {action.reasoning}*")
        if violations:
            step_lines.append(f"  ⚠️ **Policy violation:** {violations[0]}")
            hint = _violation_hint(violations)
            if hint:
                step_lines.append(f"  {hint}")
        if policy_event:
            step_lines.append(f"  🔄 **{policy_event}**")
        if "snooze_sla_penalty" in rb and rb["snooze_sla_penalty"] < 0:
            step_lines.append("  ⏰ **SLA crossed during snooze!**")
        if obs.specialist_feedback and obs.specialist_feedback not in [
            line for line in log_lines if "Specialist" in line
        ]:
            step_lines.append(f"  🧑‍💼 **Specialist:** _{obs.specialist_feedback}_")
        if repeated_action_streak == 2:
            step_lines.append(
                "  ⚠️ Rule agent exhausted its strategy — repeating fallback response. "
                "An LLM agent would vary response quality here."
            )
        if (
            not budget_warning_logged
            and rb.get("cost_spent", 0.0) > rb.get("cost_budget", 999.0)
        ):
            step_lines.append("  💸 Budget exceeded — efficiency penalty will apply at episode end.")
            budget_warning_logged = True

        log_lines.extend(step_lines)
        log_lines.append("")

        # Update scoreboard
        if components:
            component_scores = components
        if done:
            final_score = info.get("final_score", 0.0) or 0.0

        scoreboard = _build_scoreboard(
            step=step,
            max_steps=obs.max_steps,
            done=done,
            final_score=final_score,
            components=component_scores,
            cost_spent=rb.get("cost_spent", 0.0),
            cost_budget=rb.get("cost_budget", scenario.cost_budget),
            policy_version=obs.policy_version,
            failure_modes=failure_modes,
        )

        yield "\n".join(log_lines), scoreboard
        if not done:
            time.sleep(step_delay)

    # Final summary
    log_lines += [
        "---",
        "### 🏁 Episode complete",
        f"**Final score: `{final_score:.4f}`**",
        _score_verdict(final_score, scenario.difficulty),
    ]
    if component_scores:
        log_lines.append("")
        log_lines.append("**Component breakdown:**")
        for k, v in component_scores.items():
            bar = _mini_bar(v)
            log_lines.append(f"- `{k}`: {bar} `{v:.3f}`")
    ideal_summary = _ideal_agent_summary(scenario_id)
    if ideal_summary:
        log_lines.append("")
        log_lines.append("**What an ideal agent would do differently:**")
        for item in ideal_summary:
            log_lines.append(f"- {item}")

    yield "\n".join(log_lines), scoreboard


def _mini_bar(v: float, width: int = 10) -> str:
    clamped = max(0.0, min(1.0, v))
    filled = int(round(clamped * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _score_verdict(score: float, difficulty: str) -> str:
    if difficulty == "easy":
        if score >= 0.85:
            return "🏆 Excellent — rule baseline clears this tier reliably."
        if score >= 0.6:
            return "✅ Solid — basic policy compliance achieved."
        return "⚠️ Below baseline — policy or routing error occurred."
    if difficulty == "medium":
        if score >= 0.65:
            return "🏆 Strong — ambiguity handled correctly."
        if score >= 0.4:
            return "✅ Partial — clarification or response quality lacking."
        return "⚠️ Weak — likely skipped request_info on ambiguous ticket."
    # hard
    if score >= 0.7:
        return "🏆 Impressive — complex thread resolved with policy and history awareness."
    if score >= 0.45:
        return "✅ Moderate — some components missing (specialist, adversarial, history)."
    return "⚠️ Low — hard tier requires specialist coordination and adversarial reasoning."


def _ideal_agent_summary(scenario_id: str) -> list[str]:
    scenario_specific: dict[str, list[str]] = {
        "medium_adversarial_refund_already_processed": [
            "Explicitly cite prior refund evidence and avoid promising a duplicate payout.",
            "Request a transaction reference check while keeping tone firm and customer-safe.",
            "Choose escalation only if evidence is inconsistent after clarification.",
        ],
        "hard_hidden_fraud_delayed_detection": [
            "Flag fraud earlier from subtle intent signals instead of waiting for explicit confirmation.",
            "Escalate with a risk-focused rationale and communicate containment steps.",
            "Provide a concrete timeline and next action ownership in the response.",
        ],
        "hard_long_horizon_escalation_chain": [
            "Use specialist input early and keep updates structured with ownership and cadence.",
            "Avoid repetitive filler responses and progress the case each step.",
            "Balance policy compliance with budget efficiency across the full trajectory.",
        ],
    }
    default_summary = [
        "Ground responses in thread-specific facts instead of generic fallback language.",
        "Avoid repeated identical actions and adapt strategy as new signals appear.",
        "Improve clarity on timeline, ownership, and next verifiable step.",
    ]
    return scenario_specific.get(scenario_id, default_summary)


def _build_scoreboard(
    step: int,
    max_steps: int,
    done: bool,
    final_score: float,
    components: dict[str, float],
    cost_spent: float,
    cost_budget: float,
    policy_version: str,
    failure_modes: list[str],
) -> str:
    lines = ["### 📊 Live scoreboard", ""]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Steps taken | {step} / {max_steps} |")
    lines.append(f"| Cost | ${cost_spent:.3f} / ${cost_budget:.2f} |")
    lines.append(f"| Policy version | `{policy_version}` |")
    if done:
        lines.append(f"| **Final score** | **`{final_score:.4f}`** |")

    if components:
        lines.append("")
        lines.append("**Components:**")
        for k, v in components.items():
            bar = _mini_bar(v)
            lines.append(f"- `{k}`: {bar} `{v:.3f}`")

    if failure_modes:
        lines.append("")
        lines.append("**⚠️ Failure modes detected:**")
        for fm in failure_modes:
            lines.append(f"- `{fm}`")

    return "\n".join(lines)


# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Business Policy Compliance Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🏢 Business Policy Compliance & Customer Resolution Environment\n"
        "_Policy-aware agent evaluation — 43 scenarios · adversarial mechanics · specialist escalation_"
    )

    with gr.Tabs():

        # ── Tab 1: Demo ───────────────────────────────────────────────────────
        with gr.Tab("🎬 Demo — Watch the agent"):
            gr.Markdown(
                "Select a scenario and watch the **rule-based agent** solve it step by step. "
                "The agent uses keyword heuristics and policy rules — no LLM. "
                "Hard scenarios expose its limits clearly."
            )

            with gr.Row():
                demo_scenario_dd = gr.Dropdown(
                    choices=_DEMO_SCENARIOS,
                    value="hard_hidden_fraud_delayed_detection",
                    label="Scenario",
                    scale=3,
                )
                step_delay_slider = gr.Slider(
                    minimum=0.2,
                    maximum=3.0,
                    value=0.8,
                    step=0.1,
                    label="Step delay (seconds)",
                    scale=1,
                )

            with gr.Row():
                show_reasoning_cb = gr.Checkbox(
                    value=True,
                    label="Show agent reasoning",
                    scale=1,
                )
                run_demo_btn = gr.Button(
                    "▶ Run demo",
                    variant="primary",
                    scale=2,
                )
                stop_demo_btn = gr.Button(
                    "⏹ Stop",
                    variant="stop",
                    scale=1,
                )

            with gr.Row():
                demo_log = gr.Markdown(
                    label="Episode log",
                    value="_Press **Run demo** to start._",
                )
                demo_scoreboard = gr.Markdown(
                    label="Live scoreboard",
                    value="",
                )

            run_demo_btn.click(
                fn=run_demo,
                inputs=[demo_scenario_dd, step_delay_slider, show_reasoning_cb],
                outputs=[demo_log, demo_scoreboard],
            )

            gr.Markdown(
                "---\n"
                "**Difficulty guide:**  🟢 Easy — clear policy signals  |  "
                "🟡 Medium — ambiguous, requires clarification  |  "
                "🔴 Hard — multi-turn, adversarial, specialist needed"
            )

        # ── Tab 2: Manual judge UI ────────────────────────────────────────────
        with gr.Tab("🕹️ Manual — Step through yourself"):
            gr.Markdown(
                "_Select a scenario, reset the episode, then take actions one step at a time._"
            )

            with gr.Row():
                scenario_dd = gr.Dropdown(
                    choices=get_scenario_choices(),
                    label="Scenario",
                    value=None,
                )
                reset_btn = gr.Button("Reset episode", variant="primary")

            obs_display = gr.Markdown(label="Observation")

            with gr.Row():
                action_type = gr.Dropdown(
                    choices=[
                        "categorize",
                        "set_priority",
                        "draft_response",
                        "escalate",
                        "mark_spam",
                        "request_info",
                        "flag_fraud",
                        "snooze",
                        "consult_specialist",
                    ],
                    label="Action type",
                )
                reasoning = gr.Textbox(
                    label="Reasoning (not graded)",
                    placeholder="Why are you taking this action?",
                )

            with gr.Row():
                category = gr.Dropdown(
                    choices=["billing", "technical_support", "returns", "legal", "customer_success", "spam"],
                    label="Category",
                )
                priority = gr.Dropdown(
                    choices=["low", "medium", "high", "urgent"],
                    label="Priority",
                )

            with gr.Row():
                response_text = gr.Textbox(label="Response text (for draft_response)")
                escalation_reason = gr.Textbox(label="Escalation reason (for escalate)")
                clarifying_question = gr.Textbox(label="Clarifying question (for request_info)")

            with gr.Row():
                fraud_reason = gr.Textbox(label="Fraud reason (for flag_fraud)")
                snooze_hours = gr.Number(label="Snooze hours (for snooze)", precision=0, value=0)
                specialist_question = gr.Textbox(label="Specialist question (for consult_specialist)")

            step_btn = gr.Button("Take action", variant="secondary")
            scores_display = gr.Textbox(label="Component scores", interactive=False)
            status_display = gr.Textbox(label="Status / reward", interactive=False)

            reset_btn.click(
                reset_episode,
                inputs=[scenario_dd],
                outputs=[obs_display, scores_display, status_display, step_btn],
            )
            step_btn.click(
                take_action,
                inputs=[
                    action_type,
                    category,
                    priority,
                    response_text,
                    escalation_reason,
                    clarifying_question,
                    fraud_reason,
                    snooze_hours,
                    specialist_question,
                    reasoning,
                ],
                outputs=[obs_display, scores_display, status_display],
            )

        # ── Tab 3: Environment info ───────────────────────────────────────────
        with gr.Tab("📖 Environment info"):
            gr.Markdown("""
## Action space
| Action | Required field | Purpose |
|--------|---------------|---------|
| `categorize` | `category` | Route ticket to the right team |
| `set_priority` | `priority` | Set SLA urgency level |
| `draft_response` | `response_text` | Send reply to customer |
| `escalate` | `escalation_reason` | Escalate to senior team |
| `mark_spam` | — | Discard as spam |
| `request_info` | `clarifying_question` | Ask customer for clarification |
| `flag_fraud` | `fraud_reason` | Flag suspicious activity |
| `snooze` | `snooze_hours` | Defer ticket (SLA risk if >72h total) |
| `consult_specialist` | `specialist_question` | Pull in specialist review |

## Policy sets
**v1** — Refunds >$500 escalate · VIP = high/urgent · Age >72h = urgent · Legal = escalate · Suspended → billing

**v2** (v1 plus) — Premier = same-day · Fraud indicators → `flag_fraud` first

## Reward structure
| Signal | Value |
|--------|-------|
| Valid action | +0.05 |
| Policy violation | −0.20 |
| SLA crossed during snooze | −0.10 |
| Fraud missed at episode end | −0.15 |
| Efficiency bonus (≤ half steps) | +0.10 |
| Cost adjustment | ±0.12 max |
| Redundancy penalty | −0.05 per repeat |

## Baseline scores (rule agent, 43 scenarios)
| Difficulty | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Easy | 0.91 | 0.14 | 0.70 | 1.00 |
| Medium | 0.69 | 0.13 | 0.47 | 0.84 |
| Hard | 0.53 | 0.13 | 0.28 | 0.75 |

## API endpoints
`GET /health` · `GET /tasks` · `POST /reset` · `POST /step` · `GET /state` · `DELETE /session`

Session header: `X-Session-Id` (optional, defaults to `"default"`)
""")


demo.launch(server_port=7861, share=False)
