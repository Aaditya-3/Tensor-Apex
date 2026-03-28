from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["billing", "technical_support", "returns", "legal", "customer_success", "spam"]
Priority = Literal["low", "medium", "high", "urgent"]
ActionType = Literal[
    "categorize",
    "set_priority",
    "draft_response",
    "escalate",
    "mark_spam",
    "request_info",
    "flag_fraud",
    "snooze",
    "consult_specialist",
]
Difficulty = Literal["easy", "medium", "hard"]
SenderTier = Literal["standard", "vip", "premier"]
PolicyVersion = Literal["v1", "v2"]


class EpisodePhase(StrEnum):
    initial = "initial"
    awaiting_clarification = "awaiting_clarification"
    post_clarification = "post_clarification"
    resolving = "resolving"
    complete = "complete"


class EmailMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    direction: Literal["customer", "agent", "system"] = "customer"
    sender_name: str
    sender_email: str
    timestamp: datetime
    subject: str
    body: str


class TicketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    thread: list[EmailMessage]
    sender_tier: SenderTier
    account_flags: list[str] = Field(default_factory=list)
    internal_flags: list[str] = Field(default_factory=list)
    refund_amount: float | None = None
    order_id: str | None = None
    product_name: str | None = None
    visible_problem_type: str | None = None


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_category: Category | None = None
    expected_priority: Priority | None = None
    expected_escalation: bool = False
    expected_escalation_reason: str | None = None
    expected_flag_fraud: bool = False
    fraud_keywords: list[str] = Field(default_factory=list)
    requires_request_info: bool = False
    request_info_first_required: bool = False
    clarification_keywords: list[str] = Field(default_factory=list)
    response_keywords: list[str] = Field(default_factory=list)
    customer_quality_keywords: list[str] = Field(default_factory=list)
    history_keywords: list[str] = Field(default_factory=list)
    completion_action_types: list[ActionType] = Field(default_factory=list)
    ambiguous: bool = False
    requires_specialist_review: bool = False
    adversarial_pattern: str | None = None
    delayed_fraud_step_threshold: int | None = None


class TaskScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    difficulty: Difficulty
    title: str
    objective: str
    max_steps: int
    now: datetime
    policy_version: PolicyVersion = "v1"
    policy_shift_step: int | None = None
    policy_shift_to: PolicyVersion | None = None
    cost_budget: float = 0.25
    min_steps_before_completion: int = 0
    hidden_intent: Literal["honest", "fraudulent", "policy_gaming"] = "honest"
    specialist_decision: str | None = None
    adversarial_tags: list[str] = Field(default_factory=list)
    initial_snapshot: TicketSnapshot
    clarification_snapshot: TicketSnapshot | None = None
    ground_truth: GroundTruth


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    reasoning: str = Field(min_length=3)
    category: Category | None = None
    priority: Priority | None = None
    response_text: str | None = None
    escalation_reason: str | None = None
    clarifying_question: str | None = None
    fraud_reason: str | None = None
    snooze_hours: int | None = None
    specialist_question: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> Action:
        required_fields = {
            "categorize": self.category,
            "set_priority": self.priority,
            "draft_response": self.response_text,
            "escalate": self.escalation_reason,
            "request_info": self.clarifying_question,
            "flag_fraud": self.fraud_reason,
            "snooze": self.snooze_hours,
            "consult_specialist": self.specialist_question,
        }
        value = required_fields.get(self.action_type)
        if self.action_type in required_fields and value in (None, ""):
            field_name = {
                "categorize": "category",
                "set_priority": "priority",
                "draft_response": "response_text",
                "escalate": "escalation_reason",
                "request_info": "clarifying_question",
                "flag_fraud": "fraud_reason",
                "snooze": "snooze_hours",
                "consult_specialist": "specialist_question",
            }[self.action_type]
            raise ValueError(f"{field_name} is required for action_type={self.action_type}")

        if self.action_type == "snooze" and self.snooze_hours is not None and self.snooze_hours <= 0:
            raise ValueError("snooze_hours must be greater than 0")

        return self


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action: Action
    timestamp: datetime
    valid: bool = True


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    difficulty: Difficulty
    current_email: EmailMessage
    thread: list[EmailMessage]
    sender_tier: SenderTier
    account_flags: list[str]
    refund_amount: float | None = None
    issue_age_hours: float
    emails_remaining: int
    steps_taken: int
    max_steps: int
    action_history: list[ActionRecord]
    policy_rules: list[str]
    policy_version: PolicyVersion = "v1"
    policy_shift_pending: bool = False
    policy_shift_at_step: int | None = None
    policy_shift_to: PolicyVersion | None = None
    specialist_feedback: str | None = None
    task_objective: str
    clarification_received: bool = False
    episode_phase: EpisodePhase = EpisodePhase.initial


class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_action: bool
    final_score: float | None = None
    partial_score: float | None = None
    policy_violations: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    explanation: str


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: Difficulty | None = None
    scenario_id: str | None = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    components: dict[str, float]
    explanation: str
