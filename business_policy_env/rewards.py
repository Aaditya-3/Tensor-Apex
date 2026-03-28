from __future__ import annotations

from collections import Counter

from .models import Action, RewardBreakdown
from .tasks import GroundTruthPayload, component_scores, grade_actions

VALID_ACTION_REWARD = 0.05
REDUNDANT_ACTION_REWARD = 0.01
POLICY_VIOLATION_PENALTY = -0.2
POLICY_HISTORY_PENALTY = -0.08
SNOOZE_SLA_PENALTY = -0.1
FRAUD_MISSED_PENALTY = -0.15
REDUNDANCY_PENALTY = 0.05
EFFICIENCY_BONUS = 0.1
INVALID_ACTION_REWARD = -0.1
COST_EFFICIENCY_BONUS = 0.08
COST_OVER_BUDGET_PENALTY = -0.12
DELAYED_FRAUD_PENALTY = -0.12
EARLY_MISROUTE_PENALTY = -0.08


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, round(value, 4)))


def invalid_action_breakdown(message: str) -> RewardBreakdown:
    return RewardBreakdown(
        reward=_clamp_reward(INVALID_ACTION_REWARD),
        components={"invalid_action": INVALID_ACTION_REWARD},
        explanation=message,
    )


def _redundancy_penalty(action_types: list[str]) -> float:
    counts = Counter(action_types)
    repeated_actions = sum(max(0, count - 1) for count in counts.values())
    return round(repeated_actions * REDUNDANCY_PENALTY, 4)


def _step_valid_action_reward(actions: list[Action]) -> tuple[float, bool]:
    if not actions:
        return VALID_ACTION_REWARD, False
    current_action_type = actions[-1].action_type
    is_redundant = sum(1 for action in actions if action.action_type == current_action_type) > 1
    return (REDUNDANT_ACTION_REWARD if is_redundant else VALID_ACTION_REWARD), is_redundant


def _fraud_missed_penalty(actions: list[Action], fraud_expected: bool) -> float:
    if not fraud_expected:
        return 0.0
    flagged = any(action.action_type == "flag_fraud" for action in actions)
    return 0.0 if flagged else FRAUD_MISSED_PENALTY


def _cost_adjustment(action_cost: float, cost_budget: float) -> float:
    if cost_budget <= 0:
        return 0.0
    if action_cost <= cost_budget:
        efficiency_ratio = 1.0 - (action_cost / cost_budget)
        return round(efficiency_ratio * COST_EFFICIENCY_BONUS, 4)

    overspend_ratio = (action_cost - cost_budget) / cost_budget
    return round(max(COST_OVER_BUDGET_PENALTY, COST_OVER_BUDGET_PENALTY * overspend_ratio), 4)


def _running_cost_penalty(action_cost: float, cost_budget: float) -> float:
    if cost_budget <= 0 or action_cost <= cost_budget:
        return 0.0
    overspend_ratio = (action_cost - cost_budget) / cost_budget
    return round(max(-0.08, -0.08 * overspend_ratio), 4)


def _delayed_fraud_penalty(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    threshold = ground_truth["delayed_fraud_step_threshold"]
    if threshold is None or not ground_truth["expected_flag_fraud"]:
        return 0.0

    threshold_value = max(1, int(threshold))
    first_flag_index = next((idx for idx, action in enumerate(actions) if action.action_type == "flag_fraud"), None)
    if first_flag_index is None:
        return DELAYED_FRAUD_PENALTY
    steps_late = (first_flag_index + 1) - threshold_value
    if steps_late <= 0:
        return 0.0
    scaled_penalty = DELAYED_FRAUD_PENALTY * min(1.0, steps_late / threshold_value)
    return round(scaled_penalty, 4)


def _early_misroute_penalty(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    expected_category = ground_truth["expected_category"]
    if expected_category is None:
        return 0.0

    first_categorize = next((action for action in actions if action.action_type == "categorize"), None)
    if first_categorize is None:
        return 0.0
    return EARLY_MISROUTE_PENALTY if first_categorize.category != expected_category else 0.0


def shaped_reward(
    actions: list[Action],
    ground_truth: GroundTruthPayload,
    done: bool,
    max_steps: int,
    policy_violations: list[str],
    *,
    action_cost: float,
    cost_budget: float,
    snooze_crossed_sla: bool,
    fraud_expected: bool,
    policy_violation_seen: bool,
) -> RewardBreakdown:
    partial_score = grade_actions(actions, ground_truth)
    policy_penalty = POLICY_VIOLATION_PENALTY if policy_violations else 0.0
    policy_history_penalty = POLICY_HISTORY_PENALTY if done and policy_violation_seen else 0.0
    snooze_penalty = SNOOZE_SLA_PENALTY if snooze_crossed_sla else 0.0
    valid_action_reward, redundant_action = _step_valid_action_reward(actions)
    components = {
        "valid_action": valid_action_reward,
        "redundant_action_step": float(redundant_action),
        "policy_penalty": policy_penalty,
        "policy_history_penalty": policy_history_penalty,
        "snooze_sla_penalty": snooze_penalty,
        "cost_spent": round(action_cost, 4),
        "cost_budget": round(cost_budget, 4),
    }

    if done:
        efficiency_bonus = EFFICIENCY_BONUS if len(actions) <= max_steps / 2 else 0.0
        redundancy_penalty = _redundancy_penalty([action.action_type for action in actions])
        fraud_penalty = _fraud_missed_penalty(actions, fraud_expected)
        delayed_fraud_penalty = _delayed_fraud_penalty(actions, ground_truth)
        misroute_penalty = _early_misroute_penalty(actions, ground_truth)
        cost_adjustment = _cost_adjustment(action_cost, cost_budget)
        final_reward = _clamp_reward(
            partial_score
            + valid_action_reward
            + efficiency_bonus
            - redundancy_penalty
            + policy_penalty
            + policy_history_penalty
            + snooze_penalty
            + fraud_penalty
            + delayed_fraud_penalty
            + misroute_penalty
            + cost_adjustment
        )
        components.update(
            {
                "final_score": partial_score,
                "efficiency_bonus": efficiency_bonus,
                "redundancy_penalty": -redundancy_penalty,
                "fraud_missed_penalty": fraud_penalty,
                "delayed_fraud_penalty": delayed_fraud_penalty,
                "early_misroute_penalty": misroute_penalty,
                "cost_adjustment": cost_adjustment,
            }
        )
        explanation = "Final reward includes grader score, bonuses, and policy/fraud/snooze penalties."
        return RewardBreakdown(reward=final_reward, components=components, explanation=explanation)

    running_cost_penalty = _running_cost_penalty(action_cost, cost_budget)
    intermediate_reward = _clamp_reward(
        valid_action_reward + partial_score + policy_penalty + snooze_penalty + running_cost_penalty
    )
    components["partial_score"] = partial_score
    components["running_cost_penalty"] = running_cost_penalty
    explanation = "Intermediate reward includes valid-action bonus, partial score, and immediate penalties."
    return RewardBreakdown(reward=intermediate_reward, components=components, explanation=explanation)


def current_progress(actions: list[Action], ground_truth: GroundTruthPayload) -> tuple[float, dict[str, float]]:
    return grade_actions(actions, ground_truth), component_scores(actions, ground_truth)
