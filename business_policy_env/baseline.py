from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev
from typing import Any, Protocol

from .environment import BusinessPolicyComplianceEnv
from .models import Action, Category, Observation, Priority
from .tasks import scenarios_for_task


class _Agent(Protocol):
    def next_action(self, observation: Observation) -> Action: ...


class RuleBasedAgent:
    def next_action(self, observation: Observation) -> Action:
        action_types = [record.action.action_type for record in observation.action_history]
        body = observation.current_email.body.lower()
        subject = observation.current_email.subject.lower()
        combined_text = f"{subject} {body}"

        if "flag_fraud" not in action_types and self._detect_fraud(combined_text, observation.account_flags):
            return Action(
                action_type="flag_fraud",
                reasoning="Detected fraud indicators and policy requires fraud flagging first.",
                fraud_reason="Detected fraud signal from content or account risk flags.",
            )

        if (
            not observation.clarification_received
            and "request_info" not in action_types
            and self._needs_clarification(body)
        ):
            return Action(
                action_type="request_info",
                reasoning="The message is too vague to safely resolve without a follow-up question.",
                clarifying_question="Can you confirm the order or invoice involved and what outcome you want?",
            )

        if "categorize" not in action_types:
            return Action(
                action_type="categorize",
                reasoning="Route using simple keyword rules.",
                category=self._category(combined_text),
            )

        if "set_priority" not in action_types:
            return Action(
                action_type="set_priority",
                reasoning="Priority follows age and customer tier policy.",
                priority=self._priority(observation),
            )

        if "consult_specialist" not in action_types and self._needs_specialist(observation, combined_text):
            return Action(
                action_type="consult_specialist",
                reasoning="Bring in specialist context for high-risk or long-horizon tickets.",
                specialist_question="Can you confirm the safest policy-compliant next step and risk posture?",
            )

        if "escalate" not in action_types and self._needs_escalation(observation, combined_text):
            return Action(
                action_type="escalate",
                reasoning="Escalate based on refund threshold or legal language.",
                escalation_reason="Policy escalation required.",
            )

        if (
            "processed refund" in combined_text
            or "refund tx-" in combined_text
            or "processed refund tx-" in combined_text
        ):
            response_text = (
                "We reviewed the thread and found the refund was already processed. "
                "We will confirm the transaction reference and provide the settlement status."
            )
        elif any(signal in combined_text for signal in ["crash", "freez", "error", "not load"]):
            response_text = (
                "We understand the disruption. We are diagnosing the crash path now and will send "
                "device-specific steps."
            )
        elif observation.specialist_feedback:
            response_text = (
                "Thanks for your patience. We incorporated specialist guidance and are proceeding "
                "with policy-safe updates."
            )
        else:
            response_text = "We understand the delay, are reviewing this now, and will send a concrete update shortly."

        return Action(
            action_type="draft_response",
            reasoning="Send a short acknowledgement.",
            response_text=response_text,
        )

    def _detect_fraud(self, combined_text: str, account_flags: list[str]) -> bool:
        if any(flag in {"fraud_risk", "ato_watch", "chargeback_risk"} for flag in account_flags):
            return True
        signals = [
            "fraud",
            "chargeback",
            "unauthorized",
            "account takeover",
            "stolen",
            "card testing",
            "multiple cards",
            "bank reversal",
            "skip investigation",
            "bypass",
            "test transactions",
            "several new cards",
        ]
        return any(signal in combined_text for signal in signals)

    def _needs_clarification(self, body: str) -> bool:
        clear_signals = [
            "refund",
            "invoice",
            "charge",
            "payment",
            "billing",
            "login",
            "password",
            "app",
            "error",
            "update",
            "replacement",
            "return",
            "exchange",
            "fraud",
        ]
        has_clear_signal = any(signal in body for signal in clear_signals)
        return len(body.split()) < 20 and not has_clear_signal

    def _category(self, combined_text: str) -> Category:
        if any(keyword in combined_text for keyword in ["spam", "click now", "bonus", "buy fake"]):
            return "spam"
        if any(keyword in combined_text for keyword in ["lawyer", "legal action", "lawsuit", "counsel"]):
            return "legal"
        if any(keyword in combined_text for keyword in ["refund", "charge", "invoice", "payment", "billing", "card"]):
            return "billing"
        if any(keyword in combined_text for keyword in ["login", "password", "app", "error", "update", "load"]):
            return "technical_support"
        if any(keyword in combined_text for keyword in ["replacement", "return", "exchange"]):
            return "returns"
        return "customer_success"

    def _priority(self, observation: Observation) -> Priority:
        if observation.issue_age_hours > 72:
            return "urgent"
        if observation.policy_version == "v2" and observation.sender_tier == "premier":
            return "high"
        if observation.sender_tier == "vip":
            return "high"
        return "medium"

    def _needs_escalation(self, observation: Observation, combined_text: str) -> bool:
        if observation.refund_amount and observation.refund_amount > 500:
            return True
        return any(keyword in combined_text for keyword in ["lawyer", "legal action", "lawsuit"])

    def _needs_specialist(self, observation: Observation, combined_text: str) -> bool:
        objective = observation.task_objective.lower()
        if "specialist" in objective or "long-running" in objective:
            return True
        if any(keyword in combined_text for keyword in ["bypass", "card testing", "policy abuse", "legal team"]):
            return True
        return observation.sender_tier == "premier" and observation.issue_age_hours > 48


class OpenAIBaselineAgent:
    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("openai package is not installed.") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI baseline.")

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def next_action(self, observation: Observation) -> Action:  # pragma: no cover - optional path
        prompt = {
            "task_objective": observation.task_objective,
            "policy_rules": observation.policy_rules,
            "policy_version": observation.policy_version,
            "issue_age_hours": observation.issue_age_hours,
            "sender_tier": observation.sender_tier,
            "account_flags": observation.account_flags,
            "refund_amount": observation.refund_amount,
            "thread": [message.model_dump(mode="json") for message in observation.thread],
            "action_history": [record.model_dump(mode="json") for record in observation.action_history],
            "episode_phase": observation.episode_phase,
        }
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer-support policy agent. Return exactly one JSON object "
                        "that matches the Action schema."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        return Action.model_validate_json(response.output_text)


def run_episode(env: BusinessPolicyComplianceEnv, agent: _Agent, scenario_id: str) -> dict[str, Any]:
    observation = env.reset(scenario_id=scenario_id)
    reward = 0.0
    done = False
    info: dict[str, Any] = {}
    while not done:
        action = agent.next_action(observation)
        observation, reward, done, info = env.step(action)
    return {
        "scenario_id": scenario_id,
        "reward": reward,
        "final_score": info.get("final_score", 0.0),
        "component_scores": info.get("component_scores", {}),
    }


def run_baseline(agent_name: str = "rule", model: str = "gpt-4.1-mini", seed: int = 42) -> dict[str, Any]:
    env = BusinessPolicyComplianceEnv(variation_seed=seed)
    agent = RuleBasedAgent() if agent_name == "rule" else OpenAIBaselineAgent(model=model)
    summary: dict[str, Any] = {"agent": agent_name, "results": {}}

    for task_name in ["easy", "medium", "hard"]:
        task_results = [run_episode(env, agent, scenario.scenario_id) for scenario in scenarios_for_task(task_name)]
        scores = [result["final_score"] for result in task_results]
        summary["results"][task_name] = {
            "mean_final_score": round(mean(scores), 4),
            "std_final_score": round(pstdev(scores), 4),
            "min_final_score": round(min(scores), 4),
            "max_final_score": round(max(scores), 4),
            "scenarios": task_results,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline agents against the Business Policy Compliance environment."
    )
    parser.add_argument("--agent", choices=["rule", "openai"], default="rule")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_baseline(agent_name=args.agent, model=args.model, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
