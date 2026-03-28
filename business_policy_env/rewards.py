from __future__ import annotations

from collections import Counter

from .models import Action, RewardBreakdown
from .tasks import GroundTruthPayload, component_scores, grade_actions

VALID_ACTION_REWARD = 0.05
POLICY_VIOLATION_PENALTY = -0.2
SNOOZE_SLA_PENALTY = -0.1
FRAUD_MISSED_PENALTY = -0.15
REDUNDANCY_PENALTY = 0.05
EFFICIENCY_BONUS = 0.1
INVALID_ACTION_REWARD = -0.1


def _clamp_reward(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


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


def _fraud_missed_penalty(actions: list[Action], fraud_expected: bool) -> float:
    if not fraud_expected:
        return 0.0
    flagged = any(action.action_type == "flag_fraud" for action in actions)
    return 0.0 if flagged else FRAUD_MISSED_PENALTY


def shaped_reward(
    actions: list[Action],
    ground_truth: GroundTruthPayload,
    done: bool,
    max_steps: int,
    policy_violations: list[str],
    *,
    snooze_crossed_sla: bool,
    fraud_expected: bool,
) -> RewardBreakdown:
    partial_score = grade_actions(actions, ground_truth)
    policy_penalty = POLICY_VIOLATION_PENALTY if policy_violations else 0.0
    snooze_penalty = SNOOZE_SLA_PENALTY if snooze_crossed_sla else 0.0
    components = {
        "valid_action": VALID_ACTION_REWARD,
        "policy_penalty": policy_penalty,
        "snooze_sla_penalty": snooze_penalty,
    }

    if done:
        efficiency_bonus = EFFICIENCY_BONUS if len(actions) <= max_steps / 2 else 0.0
        redundancy_penalty = _redundancy_penalty([action.action_type for action in actions])
        fraud_penalty = _fraud_missed_penalty(actions, fraud_expected)
        final_reward = _clamp_reward(
            partial_score
            + VALID_ACTION_REWARD
            + efficiency_bonus
            - redundancy_penalty
            + policy_penalty
            + snooze_penalty
            + fraud_penalty
        )
        components.update(
            {
                "final_score": partial_score,
                "efficiency_bonus": efficiency_bonus,
                "redundancy_penalty": -redundancy_penalty,
                "fraud_missed_penalty": fraud_penalty,
            }
        )
        explanation = "Final reward includes grader score, bonuses, and policy/fraud/snooze penalties."
        return RewardBreakdown(reward=final_reward, components=components, explanation=explanation)

    intermediate_reward = _clamp_reward(VALID_ACTION_REWARD + partial_score + policy_penalty + snooze_penalty)
    components["partial_score"] = partial_score
    explanation = "Intermediate reward includes valid-action bonus, partial score, and immediate penalties."
    return RewardBreakdown(reward=intermediate_reward, components=components, explanation=explanation)


def current_progress(actions: list[Action], ground_truth: GroundTruthPayload) -> tuple[float, dict[str, float]]:
    return grade_actions(actions, ground_truth), component_scores(actions, ground_truth)
