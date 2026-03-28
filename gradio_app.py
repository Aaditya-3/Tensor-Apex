from __future__ import annotations

import threading
from typing import Any

import gradio as gr
import uvicorn

from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action
from business_policy_env.server import app as fastapi_app
from business_policy_env.tasks import scenario_registry


def start_api() -> None:
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="error")


threading.Thread(target=start_api, daemon=True).start()

env = BusinessPolicyComplianceEnv()
current_obs = None


def get_scenario_choices() -> list[tuple[str, str]]:
    registry = scenario_registry()
    entries = sorted(registry.values(), key=lambda item: (item.difficulty, item.scenario_id))
    return [(f"[{scenario.difficulty.upper()}] {scenario.title}", scenario.scenario_id) for scenario in entries]


def format_observation(obs) -> str:
    lines = [
        f"**Scenario:** {obs.scenario_id} [{obs.difficulty}]",
        f"**Policy version:** {obs.policy_version}",
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
        )
    except Exception as exc:  # pragma: no cover - UI validation path
        return format_observation(current_obs), "", f"Invalid action: {exc}"

    current_obs, reward, done, info = env.step(action)
    status = f"Reward: {reward:.4f} | Done: {done}"
    if info.get("policy_violations"):
        status += f" | Policy violations: {', '.join(info['policy_violations'])}"
    if done:
        status += f" | Final score: {info.get('final_score', 0):.4f}"
    return format_observation(current_obs), str(info.get("component_scores", {})), status


with gr.Blocks(title="Business Policy Compliance Environment") as demo:
    gr.Markdown("## Business Policy Compliance & Customer Resolution Environment")
    gr.Markdown("_Policy-aware agent evaluation. Select a scenario, reset, then step through actions._")

    with gr.Row():
        scenario_dd = gr.Dropdown(choices=get_scenario_choices(), label="Scenario", value=None)
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
            ],
            label="Action type",
        )
        reasoning = gr.Textbox(label="Reasoning (not graded)", placeholder="Why are you taking this action?")

    with gr.Row():
        category = gr.Dropdown(
            choices=["billing", "technical_support", "returns", "legal", "customer_success", "spam"],
            label="Category",
        )
        priority = gr.Dropdown(choices=["low", "medium", "high", "urgent"], label="Priority")

    with gr.Row():
        response_text = gr.Textbox(label="Response text (for draft_response)")
        escalation_reason = gr.Textbox(label="Escalation reason (for escalate)")
        clarifying_question = gr.Textbox(label="Clarifying question (for request_info)")

    with gr.Row():
        fraud_reason = gr.Textbox(label="Fraud reason (for flag_fraud)")
        snooze_hours = gr.Number(label="Snooze hours (for snooze)", precision=0, value=0)

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
            reasoning,
        ],
        outputs=[obs_display, scores_display, status_display],
    )

demo.launch(server_port=7861, share=False)
