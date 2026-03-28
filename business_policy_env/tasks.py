from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from .data_generation import build_scenarios
from .models import Action, TaskScenario, TicketSnapshot
from .policies import policies_satisfied

GroundTruthPayload = dict[str, Any]


def compute_issue_age_hours(snapshot: TicketSnapshot, now: datetime) -> float:
    first_timestamp = snapshot.thread[0].timestamp
    return round((now - first_timestamp).total_seconds() / 3600, 2)


@lru_cache(maxsize=1)
def scenario_registry() -> dict[str, TaskScenario]:
    return {scenario.scenario_id: scenario for scenario in build_scenarios()}


def scenarios_for_task(task_name: str | None = None) -> list[TaskScenario]:
    scenarios = list(scenario_registry().values())
    if task_name is None:
        return sorted(scenarios, key=lambda item: (item.difficulty, item.scenario_id))
    return sorted(
        [scenario for scenario in scenarios if scenario.difficulty == task_name],
        key=lambda item: item.scenario_id,
    )


def build_ground_truth_payload(scenario: TaskScenario, snapshot: TicketSnapshot) -> GroundTruthPayload:
    return {
        "difficulty": scenario.difficulty,
        "policy_version": scenario.policy_version,
        "expected_category": scenario.ground_truth.expected_category,
        "expected_priority": scenario.ground_truth.expected_priority,
        "expected_escalation": scenario.ground_truth.expected_escalation,
        "expected_escalation_reason": scenario.ground_truth.expected_escalation_reason,
        "expected_flag_fraud": scenario.ground_truth.expected_flag_fraud,
        "fraud_keywords": scenario.ground_truth.fraud_keywords,
        "requires_request_info": scenario.ground_truth.requires_request_info,
        "request_info_first_required": scenario.ground_truth.request_info_first_required,
        "clarification_keywords": scenario.ground_truth.clarification_keywords,
        "response_keywords": scenario.ground_truth.response_keywords,
        "history_keywords": scenario.ground_truth.history_keywords,
        "completion_action_types": scenario.ground_truth.completion_action_types,
        "ambiguous": scenario.ground_truth.ambiguous,
        "snapshot": snapshot.model_dump(mode="json"),
        "issue_age_hours": compute_issue_age_hours(snapshot, scenario.now),
    }


def latest_action(actions: list[Action], action_type: str) -> Action | None:
    for action in reversed(actions):
        if action.action_type == action_type:
            return action
    return None


def _request_info_quality(action: Action | None, keywords: list[str]) -> float:
    if action is None or not action.clarifying_question:
        return 0.0
    text = action.clarifying_question.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    if hits:
        return min(1.0, hits / len(keywords))
    return 0.0


def _keyword_score(text: str | None, keywords: list[str]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return min(1.0, hits / len(keywords))


def _hard_response_score(response_text: str | None, response_keywords: list[str], history_keywords: list[str]) -> float:
    if not response_text:
        return 0.0

    exact_score = _keyword_score(response_text, response_keywords + history_keywords)

    acknowledgment_signals = ["apolog", "understand", "recogni", "aware", "noted", "received"]
    timeline_signals = ["day", "hour", "week", "wait", "since", "ago", "delay", "time"]
    action_signals = ["escalat", "review", "priorit", "team", "follow", "update", "resolve", "process"]

    lowered = response_text.lower()
    ack_hit = any(signal in lowered for signal in acknowledgment_signals)
    time_hit = any(signal in lowered for signal in timeline_signals)
    action_hit = any(signal in lowered for signal in action_signals)
    semantic_score = (ack_hit + time_hit + action_hit) / 3.0

    return round(0.6 * exact_score + 0.4 * semantic_score, 4)


def _categorize_score(actions: list[Action], expected_category: str | None) -> float:
    if expected_category is None:
        return 1.0
    action = latest_action(actions, "categorize")
    if action is None:
        return 0.0
    return 1.0 if action.category == expected_category else 0.0


def _priority_score(actions: list[Action], expected_priority: str | None) -> float:
    if expected_priority is None:
        return 1.0
    action = latest_action(actions, "set_priority")
    if action is None:
        return 0.0
    return 1.0 if action.priority == expected_priority else 0.0


def _escalation_score(actions: list[Action], expected_escalation: bool) -> float:
    escalated = any(action.action_type == "escalate" for action in actions)
    return 1.0 if escalated == expected_escalation else 0.0


def _fraud_score(actions: list[Action], expected_flag_fraud: bool) -> float:
    flagged = any(action.action_type == "flag_fraud" for action in actions)
    return 1.0 if flagged == expected_flag_fraud else 0.0


def _policy_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    snapshot = TicketSnapshot.model_validate(ground_truth["snapshot"])
    return (
        1.0
        if policies_satisfied(
            actions,
            snapshot,
            float(ground_truth["issue_age_hours"]),
            ground_truth["policy_version"],
        )
        else 0.0
    )


def easy_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    components = easy_components(actions, ground_truth)
    return round(
        0.35 * components["category_correct"]
        + 0.3 * components["priority_correct"]
        + 0.2 * components["policy_compliance"]
        + 0.15 * components["fraud_handling"],
        4,
    )


def easy_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    return {
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": _policy_score(actions, ground_truth),
        "fraud_handling": _fraud_score(actions, bool(ground_truth["expected_flag_fraud"])),
    }


def medium_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    if ground_truth["request_info_first_required"]:
        if not actions or actions[0].action_type != "request_info":
            return 0.0
    components = medium_components(actions, ground_truth)
    return round(
        0.2 * components["ambiguity_recognition"]
        + 0.15 * components["clarifying_question_quality"]
        + 0.15 * components["policy_compliance"]
        + 0.1 * components["category_correct"]
        + 0.1 * components["priority_correct"]
        + 0.2 * components["response_appropriateness"]
        + 0.1 * components["fraud_handling"],
        4,
    )


def medium_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    request_info_action = actions[0] if actions and actions[0].action_type == "request_info" else None
    draft_action = latest_action(actions, "draft_response")
    return {
        "ambiguity_recognition": 1.0 if request_info_action else 0.0,
        "clarifying_question_quality": _request_info_quality(
            request_info_action,
            ground_truth["clarification_keywords"],
        ),
        "policy_compliance": _policy_score(actions, ground_truth),
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "response_appropriateness": _keyword_score(
            draft_action.response_text if draft_action else None,
            ground_truth["response_keywords"],
        ),
        "fraud_handling": _fraud_score(actions, bool(ground_truth["expected_flag_fraud"])),
    }


def hard_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    if ground_truth["request_info_first_required"]:
        if not actions or actions[0].action_type != "request_info":
            return 0.0
    components = hard_components(actions, ground_truth)
    return round(
        0.1 * components["temporal_reasoning"]
        + 0.1 * components["policy_compliance"]
        + 0.1 * components["escalation_accuracy"]
        + 0.25 * components["history_acknowledgment"]
        + 0.35 * components["response_completeness"]
        + 0.1 * components["fraud_handling"],
        4,
    )


def hard_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    draft_action = latest_action(actions, "draft_response")
    response_text = draft_action.response_text if draft_action else None
    response_keywords = _keyword_score(response_text, ground_truth["response_keywords"])
    history_score = _hard_response_score(
        response_text,
        ground_truth["response_keywords"],
        ground_truth["history_keywords"],
    )
    category_score = _categorize_score(actions, ground_truth["expected_category"])
    policy_score = 1.0 if _policy_score(actions, ground_truth) == 1.0 and category_score == 1.0 else 0.0
    return {
        "temporal_reasoning": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": policy_score,
        "escalation_accuracy": _escalation_score(actions, ground_truth["expected_escalation"]),
        "history_acknowledgment": history_score,
        "response_completeness": response_keywords,
        "fraud_handling": _fraud_score(actions, bool(ground_truth["expected_flag_fraud"])),
    }


def grade_actions(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    difficulty = ground_truth["difficulty"]
    if difficulty == "easy":
        return easy_grader(actions, ground_truth)
    if difficulty == "medium":
        return medium_grader(actions, ground_truth)
    return hard_grader(actions, ground_truth)


def component_scores(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    difficulty = ground_truth["difficulty"]
    if difficulty == "easy":
        return easy_components(actions, ground_truth)
    if difficulty == "medium":
        return medium_components(actions, ground_truth)
    return hard_components(actions, ground_truth)
