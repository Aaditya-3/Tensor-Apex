from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Any

from .data_generation import build_scenarios
from .models import Action, PolicyVersion, TaskScenario, TicketSnapshot
from .policies import policies_satisfied

GroundTruthPayload = dict[str, Any]
WORD_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "before",
    "being",
    "between",
    "could",
    "customer",
    "from",
    "have",
    "help",
    "just",
    "need",
    "please",
    "problem",
    "question",
    "regarding",
    "still",
    "support",
    "team",
    "than",
    "that",
    "there",
    "they",
    "this",
    "ticket",
    "today",
    "very",
    "want",
    "with",
    "would",
    "your",
}

SEMANTIC_KEYWORD_GROUPS: dict[str, set[str]] = {
    "refund": {"refund", "reimburse", "reimbursement", "credit", "money back"},
    "escalated": {"escalated", "escalation", "specialist", "manager review", "routing up"},
    "chargeback": {"chargeback", "bank reversal", "reversal", "dispute"},
    "investigate": {"investigate", "investigation", "review", "analyze"},
    "troubleshoot": {"troubleshoot", "debug", "diagnose", "investigate"},
    "crash": {"crash", "freezes", "hang", "stuck", "not load"},
    "support": {"support", "assist", "help"},
    "delay": {"delay", "waiting", "wait", "pending"},
    "today": {"today", "same-day", "same day", "within the day"},
    "guide": {"guide", "walkthrough", "instructions"},
    "secure": {"secure", "lock down", "protect", "contain"},
}

CATEGORY_CLAIM_TERMS: dict[str, set[str]] = {
    "billing": {"refund", "billing", "invoice", "charge", "payment", "credit"},
    "technical_support": {"crash", "error", "bug", "outage", "login", "technical"},
    "returns": {"return", "exchange", "replacement", "rma"},
    "legal": {"legal", "lawyer", "attorney", "lawsuit", "regulatory"},
    "customer_success": {"onboarding", "adoption", "training", "implementation"},
    "spam": {"spam", "phishing", "junk", "unsolicited"},
}


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


def build_ground_truth_payload(
    scenario: TaskScenario,
    snapshot: TicketSnapshot,
    *,
    policy_version: PolicyVersion | None = None,
) -> GroundTruthPayload:
    resolved_policy = policy_version or scenario.policy_version
    return {
        "difficulty": scenario.difficulty,
        "policy_version": resolved_policy,
        "base_policy_version": scenario.policy_version,
        "policy_shift_step": scenario.policy_shift_step,
        "policy_shift_to": scenario.policy_shift_to,
        "cost_budget": scenario.cost_budget,
        "min_steps_before_completion": scenario.min_steps_before_completion,
        "hidden_intent": scenario.hidden_intent,
        "adversarial_tags": scenario.adversarial_tags,
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
        "customer_quality_keywords": scenario.ground_truth.customer_quality_keywords,
        "history_keywords": scenario.ground_truth.history_keywords,
        "completion_action_types": scenario.ground_truth.completion_action_types,
        "ambiguous": scenario.ground_truth.ambiguous,
        "requires_specialist_review": scenario.ground_truth.requires_specialist_review,
        "adversarial_pattern": scenario.ground_truth.adversarial_pattern,
        "delayed_fraud_step_threshold": scenario.ground_truth.delayed_fraud_step_threshold,
        "snapshot": snapshot.model_dump(mode="json"),
        "issue_age_hours": compute_issue_age_hours(snapshot, scenario.now),
    }


def latest_action(actions: list[Action], action_type: str) -> Action | None:
    for action in reversed(actions):
        if action.action_type == action_type:
            return action
    return None


def _normalized_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def _content_terms(text: str) -> list[str]:
    return [token for token in _normalized_tokens(text) if len(token) > 3 and token not in STOPWORDS]


def _semantic_variants(keyword: str) -> set[str]:
    lowered = keyword.lower()
    variants = {lowered}
    for canonical, group in SEMANTIC_KEYWORD_GROUPS.items():
        if lowered == canonical or lowered in group or canonical in lowered or lowered in canonical:
            variants.update(group)
            variants.add(canonical)
    return variants


def _contains_keyword_signal(text: str, keyword: str) -> bool:
    lowered = text.lower()
    variants = _semantic_variants(keyword)
    if any(variant in lowered for variant in variants):
        return True

    phrase_tokens = [token for token in _normalized_tokens(keyword) if len(token) > 2]
    if not phrase_tokens:
        return False
    text_tokens = set(_normalized_tokens(text))
    return all(token in text_tokens for token in phrase_tokens)


def _keyword_stuffing_penalty(text: str) -> float:
    tokens = [token for token in _normalized_tokens(text) if len(token) > 2]
    if len(tokens) < 12:
        return 0.0

    counts = Counter(tokens)
    unique_ratio = len(counts) / len(tokens)
    max_repeat_ratio = max(counts.values()) / len(tokens)
    penalty = 0.0

    if unique_ratio < 0.45:
        penalty += min(0.3, (0.45 - unique_ratio) * 0.9)
    if max_repeat_ratio > 0.2:
        penalty += min(0.2, (max_repeat_ratio - 0.2) * 1.6)

    return round(min(0.45, penalty), 4)


def _thread_focus_terms(ground_truth: GroundTruthPayload, *, max_terms: int = 12) -> list[str]:
    snapshot = ground_truth.get("snapshot", {})
    if not isinstance(snapshot, dict):
        return []

    thread = snapshot.get("thread", [])
    if not isinstance(thread, list):
        return []

    customer_messages = [
        str(message.get("body", ""))
        for message in thread
        if isinstance(message, dict) and message.get("direction") == "customer"
    ]
    if not customer_messages:
        return []

    source_text = " ".join(customer_messages[-2:])
    counts = Counter(_content_terms(source_text))
    generic_terms = {
        "account",
        "billing",
        "issue",
        "message",
        "reply",
        "request",
        "service",
        "status",
        "update",
        "urgent",
    }
    return [token for token, _ in counts.most_common() if token not in generic_terms][:max_terms]


def _thread_grounding_score(response_text: str | None, ground_truth: GroundTruthPayload) -> float:
    if not response_text:
        return 0.0
    focus_terms = _thread_focus_terms(ground_truth)
    if not focus_terms:
        return 0.5

    response_terms = set(_content_terms(response_text))
    hits = sum(1 for term in focus_terms if term in response_terms)
    target = min(5, len(focus_terms))
    return round(min(1.0, hits / max(1, target)), 4)


def _response_structure_score(response_text: str | None) -> float:
    if not response_text:
        return 0.0
    lowered = response_text.lower()
    timeline = any(signal in lowered for signal in ["today", "within", "hour", "day", "timeline", "next step"])
    ownership = any(signal in lowered for signal in ["we ", "our team", "assigned", "owner", "i will"])
    forward_plan = any(
        signal in lowered
        for signal in ["will", "follow up", "investigat", "review", "update", "escalat", "diagnos", "confirm"]
    )
    return round((float(timeline) + float(ownership) + float(forward_plan)) / 3.0, 4)


def _response_action_consistency_score(actions: list[Action], response_text: str | None) -> float:
    if not response_text:
        return 0.0

    lowered = response_text.lower()
    escalated = any(action.action_type == "escalate" for action in actions)
    consulted = any(action.action_type == "consult_specialist" for action in actions)
    fraud_flagged = any(action.action_type == "flag_fraud" for action in actions)

    penalty = 0.0
    if any(phrase in lowered for phrase in ["already escalated", "has been escalated", "escalated this"]):
        penalty += 0.35 if not escalated else 0.0
    if any(phrase in lowered for phrase in ["specialist confirmed", "after specialist review", "risk team confirmed"]):
        penalty += 0.35 if not consulted else 0.0
    if any(phrase in lowered for phrase in ["fraud was flagged", "flagged this as fraud", "already flagged fraud"]):
        penalty += 0.35 if not fraud_flagged else 0.0
    category_action = latest_action(actions, "categorize")
    claimed_category = _response_claimed_category(response_text)
    if category_action is not None and category_action.category is not None:
        if category_action.category == "technical_support":
            billing_claim = any(
                phrase in lowered for phrase in ["process refund", "issue refund", "refund approved"]
            )
            penalty += 0.3 if billing_claim else 0.0
        if claimed_category is not None and claimed_category != category_action.category:
            penalty += 0.25

    return round(max(0.0, 1.0 - penalty), 4)


def _response_claimed_category(response_text: str) -> str | None:
    lowered = response_text.lower()
    best_category: str | None = None
    best_hits = 0
    for category, terms in CATEGORY_CLAIM_TERMS.items():
        hits = sum(1 for term in terms if term in lowered)
        threshold = 1 if category in {"legal", "spam"} else 2
        if hits >= threshold and hits > best_hits:
            best_category = category
            best_hits = hits
    return best_category


def _hybrid_response_score(
    response_text: str | None,
    keywords: list[str],
    ground_truth: GroundTruthPayload,
    actions: list[Action],
) -> float:
    if not response_text:
        return 0.0
    keyword_score = _keyword_score(response_text, keywords)
    thread_grounding = _thread_grounding_score(response_text, ground_truth)
    structure = _response_structure_score(response_text)
    consistency = _response_action_consistency_score(actions, response_text)
    return round(
        min(1.0, 0.35 * keyword_score + 0.3 * thread_grounding + 0.2 * structure + 0.15 * consistency),
        4,
    )


def _ambiguity_recognition_score(actions: list[Action], request_info_first_required: bool) -> float:
    if not actions:
        return 0.0

    request_info_index = next((idx for idx, action in enumerate(actions) if action.action_type == "request_info"), None)
    if request_info_index is None:
        return 0.0
    if request_info_index == 0:
        return 1.0
    if request_info_first_required and request_info_index == 1:
        return 0.7
    if request_info_first_required:
        return 0.0
    return 0.7


def _request_info_quality(action: Action | None, keywords: list[str]) -> float:
    if action is None or not action.clarifying_question:
        return 0.0
    text = action.clarifying_question
    keyword_match = _keyword_score(text, keywords)
    lowered = text.lower()
    question_form = 1.0 if "?" in text else 0.4
    specificity = 1.0 if any(token in lowered for token in ["order", "invoice", "account", "error", "amount"]) else 0.5
    return round(min(1.0, 0.75 * keyword_match + 0.15 * question_form + 0.1 * specificity), 4)


def _keyword_score(text: str | None, keywords: list[str]) -> float:
    if not text:
        return 0.0
    if not keywords:
        return round(max(0.0, 1.0 - _keyword_stuffing_penalty(text)), 4)
    hits = sum(1 for keyword in keywords if _contains_keyword_signal(text, keyword))
    base_score = min(1.0, hits / len(keywords))
    token_count = len(_normalized_tokens(text))
    density_penalty = 0.0
    if token_count > 80:
        expected_hits = max(1, token_count // 20)
        if hits < expected_hits:
            density_penalty = min(0.4, (expected_hits - hits) * 0.08)
    penalty = min(0.75, _keyword_stuffing_penalty(text) + density_penalty)
    return round(max(0.0, base_score * (1.0 - penalty)), 4)


def _hard_response_score(
    response_text: str | None,
    response_keywords: list[str],
    history_keywords: list[str],
    ground_truth: GroundTruthPayload,
) -> float:
    if not response_text:
        return 0.0

    exact_score = _keyword_score(response_text, response_keywords + history_keywords)
    history_score = _keyword_score(response_text, history_keywords)

    acknowledgment_signals = ["apolog", "understand", "recogni", "aware", "noted", "received"]
    timeline_signals = ["day", "hour", "week", "wait", "since", "ago", "delay", "timeline"]
    action_signals = ["escalat", "review", "priorit", "team", "follow", "update", "resolve", "process"]

    lowered = response_text.lower()
    ack_hit = any(signal in lowered for signal in acknowledgment_signals)
    time_hit = any(signal in lowered for signal in timeline_signals)
    action_hit = any(signal in lowered for signal in action_signals)
    semantic_score = ((ack_hit + time_hit + action_hit) / 3.0) * history_score
    thread_grounding = _thread_grounding_score(response_text, ground_truth)

    return round(0.55 * exact_score + 0.2 * history_score + 0.15 * semantic_score + 0.1 * thread_grounding, 4)


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


def _fraud_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    expected_flag_fraud = bool(ground_truth["expected_flag_fraud"])
    first_flag_index = next((idx for idx, action in enumerate(actions) if action.action_type == "flag_fraud"), None)
    flagged = first_flag_index is not None
    if not expected_flag_fraud:
        return 1.0 if not flagged else 0.0
    if not flagged:
        return 0.0

    threshold = ground_truth["delayed_fraud_step_threshold"]
    if threshold is None:
        return 1.0
    threshold_value = int(threshold)
    if threshold_value <= 0:
        return 1.0

    assert first_flag_index is not None  # narrowed by flagged guard above
    detection_step = first_flag_index + 1
    if detection_step <= threshold_value:
        return 1.0
    return round(max(0.0, threshold_value / detection_step), 4)


def _specialist_score(actions: list[Action], requires_specialist_review: bool) -> float:
    consulted = any(action.action_type == "consult_specialist" for action in actions)
    return 1.0 if consulted == requires_specialist_review else 0.0


def _customer_quality_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    draft_action = latest_action(actions, "draft_response")
    if draft_action is None or not draft_action.response_text:
        return 0.0

    keywords = ground_truth["customer_quality_keywords"] or ground_truth["response_keywords"]
    if not keywords:
        lowered = draft_action.response_text.lower()
        empathy = any(token in lowered for token in ["sorry", "understand", "frustrat", "apolog"])
        next_steps = any(token in lowered for token in ["next", "update", "timeline", "follow", "review"])
        ownership = any(token in lowered for token in ["we will", "our team", "assigned", "investigat", "resolve"])
        base_quality = round((empathy + next_steps + ownership) / 3.0, 4)
    else:
        base_quality = _keyword_score(draft_action.response_text, keywords)

    grounding = _thread_grounding_score(draft_action.response_text, ground_truth)
    consistency = _response_action_consistency_score(actions, draft_action.response_text)
    return round(min(1.0, 0.5 * base_quality + 0.3 * grounding + 0.2 * consistency), 4)


def _clarification_strategy_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    if not ground_truth["requires_request_info"]:
        return 1.0
    request_info_action = next((action for action in actions if action.action_type == "request_info"), None)
    recognition = _ambiguity_recognition_score(actions, bool(ground_truth["request_info_first_required"]))
    quality = _request_info_quality(request_info_action, ground_truth["clarification_keywords"])
    return round(0.6 * recognition + 0.4 * quality, 4)


def _adversarial_consistency_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    pattern = ground_truth["adversarial_pattern"]
    if not pattern:
        return 1.0

    drafted = latest_action(actions, "draft_response")
    response_text = drafted.response_text if drafted else ""
    escalated = any(action.action_type == "escalate" for action in actions)
    flagged = any(action.action_type == "flag_fraud" for action in actions)
    consulted = any(action.action_type == "consult_specialist" for action in actions)
    category_action = latest_action(actions, "categorize")
    category = category_action.category if category_action else None

    if pattern == "refund_already_processed":
        acknowledgment = _keyword_score(response_text, ["already", "processed", "confirmation", "reference"])
        no_duplicate_refund_push = 1.0 if not escalated else 0.0
        return round(0.7 * acknowledgment + 0.3 * no_duplicate_refund_push, 4)

    if pattern == "policy_gaming_refund_pressure":
        return round((0.4 * float(flagged)) + (0.3 * float(escalated)) + (0.3 * float(consulted)), 4)

    if pattern in {"keyword_refund_trap", "sarcastic_mixed_intent"}:
        correct_route = 1.0 if category == "technical_support" else 0.0
        response_focus = _keyword_score(response_text, ["crash", "support", "diagnose"])
        return round(0.6 * correct_route + 0.4 * response_focus, 4)

    if pattern == "hidden_fraud_signal":
        fraud_timeliness = _fraud_score(actions, ground_truth)
        return round((0.5 * fraud_timeliness) + (0.3 * float(escalated)) + (0.2 * float(consulted)), 4)

    if pattern == "delayed_escalation_chain":
        timeline_quality = _keyword_score(response_text, ["timeline", "update", "ownership"])
        return round((0.35 * float(consulted)) + (0.35 * float(escalated)) + (0.3 * timeline_quality), 4)

    return 1.0


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
        "fraud_handling": _fraud_score(actions, ground_truth),
    }


def medium_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    components = medium_components(actions, ground_truth)
    return round(
        0.16 * components["ambiguity_recognition"]
        + 0.12 * components["clarifying_question_quality"]
        + 0.15 * components["policy_compliance"]
        + 0.1 * components["category_correct"]
        + 0.1 * components["priority_correct"]
        + 0.14 * components["response_appropriateness"]
        + 0.08 * components["fraud_handling"]
        + 0.08 * components["specialist_coordination"]
        + 0.07 * components["adversarial_resilience"],
        4,
    )


def medium_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    request_info_action = next((action for action in actions if action.action_type == "request_info"), None)
    draft_action = latest_action(actions, "draft_response")
    return {
        "ambiguity_recognition": _ambiguity_recognition_score(
            actions,
            bool(ground_truth["request_info_first_required"]),
        ),
        "clarifying_question_quality": _request_info_quality(
            request_info_action,
            ground_truth["clarification_keywords"],
        ),
        "policy_compliance": _policy_score(actions, ground_truth),
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "response_appropriateness": _hybrid_response_score(
            draft_action.response_text if draft_action else None,
            ground_truth["response_keywords"],
            ground_truth,
            actions,
        ),
        "fraud_handling": _fraud_score(actions, ground_truth),
        "specialist_coordination": _specialist_score(actions, bool(ground_truth["requires_specialist_review"])),
        "adversarial_resilience": _adversarial_consistency_score(actions, ground_truth),
    }


def hard_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    components = hard_components(actions, ground_truth)
    return round(
        0.04 * components["temporal_reasoning"]
        + 0.06 * components["policy_compliance"]
        + 0.1 * components["escalation_accuracy"]
        + 0.14 * components["history_acknowledgment"]
        + 0.1 * components["response_completeness"]
        + 0.1 * components["fraud_handling"]
        + 0.15 * components["specialist_coordination"]
        + 0.18 * components["adversarial_resilience"]
        + 0.07 * components["customer_quality"]
        + 0.06 * components["clarification_strategy"],
        4,
    )


def hard_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    draft_action = latest_action(actions, "draft_response")
    response_text = draft_action.response_text if draft_action else None
    response_keywords = _hybrid_response_score(response_text, ground_truth["response_keywords"], ground_truth, actions)
    history_score = _hard_response_score(
        response_text,
        ground_truth["response_keywords"],
        ground_truth["history_keywords"],
        ground_truth,
    )
    category_score = _categorize_score(actions, ground_truth["expected_category"])
    policy_score = 1.0 if _policy_score(actions, ground_truth) == 1.0 and category_score == 1.0 else 0.0
    return {
        "temporal_reasoning": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": policy_score,
        "escalation_accuracy": _escalation_score(actions, ground_truth["expected_escalation"]),
        "history_acknowledgment": history_score,
        "response_completeness": response_keywords,
        "fraud_handling": _fraud_score(actions, ground_truth),
        "specialist_coordination": _specialist_score(actions, bool(ground_truth["requires_specialist_review"])),
        "adversarial_resilience": _adversarial_consistency_score(actions, ground_truth),
        "customer_quality": _customer_quality_score(actions, ground_truth),
        "clarification_strategy": _clarification_strategy_score(actions, ground_truth),
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


def evaluation_metrics(
    actions: list[Action],
    ground_truth: GroundTruthPayload,
    *,
    max_steps: int,
    action_cost: float,
    cost_budget: float,
    policy_violation_seen: bool = False,
) -> dict[str, float]:
    score = grade_actions(actions, ground_truth)
    policy_score = _policy_score(actions, ground_truth)
    if policy_violation_seen:
        policy_score = min(policy_score, 0.7)
    fraud_score = _fraud_score(actions, ground_truth)
    escalation_score = _escalation_score(actions, bool(ground_truth["expected_escalation"]))
    specialist_score = _specialist_score(actions, bool(ground_truth["requires_specialist_review"]))
    adversarial_resilience = _adversarial_consistency_score(actions, ground_truth)
    customer_quality = _customer_quality_score(actions, ground_truth)
    efficiency = 1.0 if cost_budget <= 0 else max(0.0, min(1.0, 1.0 - (action_cost / cost_budget)))
    latency = max(0.0, min(1.0, 1.0 - (len(actions) / max_steps))) if max_steps > 0 else 0.0

    risk_management = round(
        min(1.0, 0.45 * fraud_score + 0.35 * escalation_score + 0.2 * specialist_score),
        4,
    )
    return {
        "score": round(score, 4),
        "policy_score": round(policy_score, 4),
        "efficiency": round(efficiency, 4),
        "latency": round(latency, 4),
        "customer_quality": round(customer_quality, 4),
        "risk_management": risk_management,
        "adversarial_resilience": round(adversarial_resilience, 4),
    }


def failure_modes(
    metrics: dict[str, float],
    *,
    policy_violations: list[str],
    done: bool,
) -> list[str]:
    failures: list[str] = []
    if policy_violations:
        failures.append("policy_violation")
    if metrics["adversarial_resilience"] < 0.55:
        failures.append("adversarial_miss")
    if metrics["risk_management"] < 0.6:
        failures.append("risk_handling_gap")
    if metrics["customer_quality"] < 0.55:
        failures.append("customer_communication_low")
    if metrics["efficiency"] < 0.25:
        failures.append("high_operational_cost")
    if done and metrics["latency"] < 0.2:
        failures.append("slow_resolution")
    return failures
