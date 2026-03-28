from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from .models import Action, Category, PolicyVersion, Priority, TicketSnapshot

POLICY_SETS: dict[str, list[str]] = {
    "v1": [
        "Refunds over $500 require escalation.",
        "VIP customers require high or urgent priority.",
        "Issues open over 72 hours require urgent priority.",
        "Complaints mentioning legal action or lawsuits require immediate escalation.",
        "Suspended accounts must be routed to the billing team.",
    ],
    "v2": [
        "Refunds over $500 require escalation.",
        "VIP customers require high or urgent priority.",
        "Issues open over 72 hours require urgent priority.",
        "Complaints mentioning legal action or lawsuits require immediate escalation.",
        "Suspended accounts must be routed to the billing team.",
        "Premier accounts require same-day response regardless of issue type.",
        "Fraud indicators require flag_fraud before any resolution action.",
    ],
}

PRIORITY_ORDER: dict[Priority, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "urgent": 3,
}

FRAUD_SIGNALS = [
    "chargeback",
    "stolen",
    "unauthorized",
    "card testing",
    "account takeover",
    "fraud",
    "suspicious",
    "multiple cards",
    "bank reversal",
    "identity theft",
]

RESOLUTION_ACTION_TYPES = {"categorize", "set_priority", "draft_response", "escalate", "mark_spam"}


class PolicyExpectations(TypedDict):
    requires_escalation: bool
    requires_fraud_flag: bool
    min_priority: Priority | None
    forced_category: Category | None
    triggered_rules: list[str]


def policy_rules_for(version: PolicyVersion) -> list[str]:
    return POLICY_SETS[version]


def has_fraud_indicators(snapshot: TicketSnapshot) -> bool:
    body_text = " ".join(message.body.lower() for message in snapshot.thread)
    flags = snapshot.account_flags + snapshot.internal_flags
    if any(flag in {"fraud_risk", "ato_watch", "chargeback_risk"} for flag in flags):
        return True
    return any(signal in body_text for signal in FRAUD_SIGNALS)


def compute_policy_expectations(
    snapshot: TicketSnapshot,
    issue_age_hours: float,
    policy_version: PolicyVersion,
) -> PolicyExpectations:
    latest_message = snapshot.thread[-1]
    body_text = latest_message.body.lower()
    expectations: PolicyExpectations = {
        "requires_escalation": False,
        "requires_fraud_flag": False,
        "min_priority": None,
        "forced_category": None,
        "triggered_rules": [],
    }

    if snapshot.refund_amount and snapshot.refund_amount > 500:
        expectations["requires_escalation"] = True
        expectations["triggered_rules"].append("refund_gt_500")

    if snapshot.sender_tier == "vip":
        expectations["min_priority"] = "high"
        expectations["triggered_rules"].append("vip_priority")

    if issue_age_hours > 72:
        expectations["min_priority"] = "urgent"
        expectations["triggered_rules"].append("sla_breach")

    if "lawsuit" in body_text or "legal action" in body_text or "lawyer" in body_text:
        expectations["requires_escalation"] = True
        expectations["triggered_rules"].append("legal_threat")

    if "suspended" in snapshot.account_flags:
        expectations["forced_category"] = "billing"
        expectations["triggered_rules"].append("suspended_to_billing")

    if policy_version == "v2" and snapshot.sender_tier == "premier":
        if issue_age_hours > 24:
            expectations["min_priority"] = "urgent"
        elif (
            expectations["min_priority"] is None
            or PRIORITY_ORDER[expectations["min_priority"]] < PRIORITY_ORDER["high"]
        ):
            expectations["min_priority"] = "high"
        expectations["triggered_rules"].append("premier_same_day")

    if policy_version == "v2" and has_fraud_indicators(snapshot):
        expectations["requires_fraud_flag"] = True
        expectations["triggered_rules"].append("fraud_flag_required")

    return expectations


def check_policy_violations(
    action: Action,
    snapshot: TicketSnapshot,
    issue_age_hours: float,
    policy_version: PolicyVersion,
    prior_actions: Iterable[Action] | None = None,
) -> list[str]:
    expectations = compute_policy_expectations(snapshot, issue_age_hours, policy_version)
    prior_actions = list(prior_actions or [])
    violations: list[str] = []

    min_priority = expectations["min_priority"]
    if action.action_type == "set_priority" and min_priority:
        if action.priority is None:
            violations.append("Priority value is required for set_priority actions.")
        elif PRIORITY_ORDER[action.priority] < PRIORITY_ORDER[min_priority]:
            violations.append(f"Priority must be at least {min_priority}.")

    if action.action_type == "categorize" and expectations["forced_category"]:
        if action.category != expectations["forced_category"]:
            violations.append(f"Category must be {expectations['forced_category']}.")

    escalated_already = any(item.action_type == "escalate" for item in prior_actions)
    if (
        action.action_type in {"draft_response", "mark_spam"}
        and expectations["requires_escalation"]
        and not escalated_already
    ):
        violations.append("The ticket requires escalation before resolution.")

    fraud_flagged_already = any(item.action_type == "flag_fraud" for item in prior_actions)
    if (
        action.action_type in RESOLUTION_ACTION_TYPES
        and expectations["requires_fraud_flag"]
        and action.action_type != "flag_fraud"
        and not fraud_flagged_already
    ):
        violations.append("Fraud indicators require flag_fraud before resolution actions.")

    return violations


def policies_satisfied(
    actions: Iterable[Action],
    snapshot: TicketSnapshot,
    issue_age_hours: float,
    policy_version: PolicyVersion,
) -> bool:
    expectations = compute_policy_expectations(snapshot, issue_age_hours, policy_version)
    latest_priority = None
    latest_category = None
    escalated = False
    flagged_fraud = False

    for action in actions:
        if action.action_type == "set_priority":
            latest_priority = action.priority
        elif action.action_type == "categorize":
            latest_category = action.category
        elif action.action_type == "escalate":
            escalated = True
        elif action.action_type == "flag_fraud":
            flagged_fraud = True

    min_priority = expectations["min_priority"]
    if min_priority:
        if latest_priority is None:
            return False
        if PRIORITY_ORDER[latest_priority] < PRIORITY_ORDER[min_priority]:
            return False

    forced_category = expectations["forced_category"]
    if forced_category and latest_category != forced_category:
        return False

    if expectations["requires_escalation"] and not escalated:
        return False

    if expectations["requires_fraud_flag"]:
        if not flagged_fraud:
            return False
        first_resolution_index = next(
            (i for i, action in enumerate(actions) if action.action_type in RESOLUTION_ACTION_TYPES),
            None,
        )
        first_fraud_flag_index = next(
            (i for i, action in enumerate(actions) if action.action_type == "flag_fraud"),
            None,
        )
        if (
            first_resolution_index is not None
            and first_fraud_flag_index is not None
            and first_fraud_flag_index > first_resolution_index
        ):
            return False

    return True
