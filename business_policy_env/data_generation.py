from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from faker import Faker

from .models import (
    ActionType,
    Category,
    Difficulty,
    EmailMessage,
    GroundTruth,
    PolicyVersion,
    Priority,
    SenderTier,
    TaskScenario,
    TicketSnapshot,
)


@dataclass
class ScenarioTemplate:
    scenario_id: str
    title: str
    subject: str
    difficulty: Difficulty
    sender_tier: SenderTier
    account_flags: list[str]
    refund_amount: float | None
    age_hours: float
    thread_bodies: list[str]
    expected_category: Category
    expected_priority: Priority
    expected_escalation: bool
    requires_request_info: bool
    request_info_first_required: bool
    response_keywords: list[str]
    history_keywords: list[str]
    clarification_keywords: list[str]
    internal_flags: list[str] = field(default_factory=list)
    hide_fraud_signals: bool = False
    clarification_body: str | None = None
    legal_language: bool = False
    suspended_account: bool = False
    expected_flag_fraud: bool = False
    fraud_keywords: list[str] = field(default_factory=list)
    policy_version: PolicyVersion = "v1"
    policy_shift_step: int | None = None
    policy_shift_to: PolicyVersion | None = None
    cost_budget: float | None = None
    min_steps_before_completion: int = 0
    hidden_intent: Literal["honest", "fraudulent", "policy_gaming"] = "honest"
    specialist_decision: str | None = None
    requires_specialist_review: bool = False
    adversarial_tags: list[str] = field(default_factory=list)
    adversarial_pattern: str | None = None
    delayed_fraud_step_threshold: int | None = None
    customer_quality_keywords: list[str] = field(default_factory=list)
    style_noise: bool = False
    emotional_tone: bool = False
    max_steps: int = 6
    objective: str | None = None
    expected_escalation_reason: str | None = None
    thread_directions: list[Literal["customer", "agent", "system"]] = field(default_factory=list)
    visible_problem_type: str | None = None


EASY_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="easy_vip_refund",
        title="VIP refund request over threshold",
        subject="Refund for duplicate annual charge",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=650.0,
        age_hours=18,
        thread_bodies=["I was charged twice and need the $650 duplicate payment refunded."],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_suspended_account",
        title="Suspended account billing route",
        subject="Why is my payment access blocked?",
        difficulty="easy",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=9,
        thread_bodies=["My account was suspended after yesterday's invoice and I need billing help."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        suspended_account=True,
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_legal_threat",
        title="Explicit legal escalation",
        subject="Final warning before legal action",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=120.0,
        age_hours=6,
        thread_bodies=["If this charge is not corrected today, I will take legal action and contact my lawyer."],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        legal_language=True,
        expected_escalation_reason="Customer mentioned legal action.",
        max_steps=5,
        visible_problem_type="legal",
    ),
    ScenarioTemplate(
        scenario_id="easy_sla_breach",
        title="Aged low-tone billing ticket",
        subject="Quick question about the March invoice",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=132,
        thread_bodies=["Could you explain the service adjustment listed on line 4 of my invoice?"],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_standard_small_refund",
        title="Standard refund below escalation threshold",
        subject="Please refund an accidental duplicate add-on",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=85.0,
        age_hours=16,
        thread_bodies=["I accidentally bought the same add-on twice and need the extra $85 refunded."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_vip_technical_issue",
        title="VIP technical issue without escalation",
        subject="Desktop app crash after patch",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=None,
        age_hours=7,
        thread_bodies=["Our desktop app crashes right after login since today's patch."],
        expected_category="technical_support",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="technical_support",
    ),
    ScenarioTemplate(
        scenario_id="easy_vip_legal_dual_trigger",
        title="VIP legal threat with overlapping rules",
        subject="Counsel review if this is not corrected today",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=210.0,
        age_hours=28,
        thread_bodies=[
            "I am a VIP account holder and our legal counsel will proceed if this charge remains unresolved."
        ],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        legal_language=True,
        expected_escalation_reason="Customer mentioned legal action.",
        max_steps=5,
        visible_problem_type="legal",
    ),
    ScenarioTemplate(
        scenario_id="easy_suspended_sla_breach",
        title="Suspended account and SLA breach",
        subject="Still locked out of billing controls",
        difficulty="easy",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=88,
        thread_bodies=["My account is still suspended and I cannot update billing settings after several days."],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        suspended_account=True,
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_spam_detection",
        title="Obvious spam outreach",
        subject="Guaranteed growth hack click now",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=2,
        thread_bodies=["Buy fake reviews and guaranteed traffic now. Click this short link to activate your bonus."],
        expected_category="spam",
        expected_priority="low",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="spam",
    ),
    ScenarioTemplate(
        scenario_id="easy_sla_marginal",
        title="Clean baseline close to SLA threshold",
        subject="Need onboarding help",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=71,
        thread_bodies=["Could someone point me to the right setup guide for our new workspace?"],
        expected_category="customer_success",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="customer_success",
    ),
]


MEDIUM_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="medium_charge_or_bug",
        title="Ambiguous charge issue for VIP customer",
        subject="The charge issue is still happening",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        refund_amount=720.0,
        age_hours=14,
        thread_bodies=["Can you fix the charge problem on my account?"],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "escalated", "duplicate"],
        history_keywords=["charge", "problem"],
        clarification_keywords=["charge", "invoice", "account"],
        clarification_body="It is the duplicate renewal charge from invoice INV-882. Please refund the extra $720.",
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_same_problem",
        title="Ambiguous repeated problem without visible history",
        subject="Same problem as last time",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=11,
        thread_bodies=["I am hitting the same problem again."],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["troubleshoot", "app", "update"],
        history_keywords=["same problem", "again"],
        clarification_keywords=["problem", "last time", "error"],
        clarification_body="The desktop app will not load past the spinning logo after the update.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_exchange_or_refund",
        title="Return request that becomes a high-value refund",
        subject="Need help with the laptop order",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=680.0,
        age_hours=90,
        thread_bodies=["I need help with my laptop order because the same issue keeps coming back."],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "escalated", "wait"],
        history_keywords=["issue", "coming back"],
        clarification_keywords=["order", "refund", "replacement"],
        clarification_body="I do not want another replacement. I need a refund for the $680 device instead.",
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_vip_small_refund_after_clarification",
        title="Ambiguous VIP complaint resolved below escalation threshold",
        subject="You charged us incorrectly again",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        refund_amount=130.0,
        age_hours=22,
        thread_bodies=["This charge problem happened again and we need it corrected."],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "invoice", "confirmed"],
        history_keywords=["again", "charge"],
        clarification_keywords=["amount", "invoice", "outcome"],
        clarification_body="It is a duplicate $130 line item on invoice INV-909 and a refund is all we need.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_returns_exchange_ambiguity",
        title="Ambiguous return versus exchange request",
        subject="Need a different size",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=19,
        thread_bodies=["The jacket did not fit and I need a different option."],
        expected_category="returns",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["exchange", "size", "return label"],
        history_keywords=["different option"],
        clarification_keywords=["refund", "exchange", "size"],
        clarification_body="Please exchange it for one size up; I do not need a refund.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_suspended_tech_vs_billing",
        title="Suspended account with technical-vs-billing ambiguity",
        subject="Cannot sign in to manage my account",
        difficulty="medium",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=26,
        thread_bodies=["I cannot log in and my account is unusable right now."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["billing", "payment", "suspended"],
        history_keywords=["cannot log in"],
        clarification_keywords=["payment", "invoice", "suspended"],
        clarification_body="The login issue started after a failed invoice payment and the account was suspended.",
        suspended_account=True,
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_multilingual_signal",
        title="Non-English message with weak keyword overlap",
        subject="Necesito ayuda urgente",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=13,
        thread_bodies=["Necesito ayuda con mi cuenta, no funciona desde ayer."],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["app", "access", "support"],
        history_keywords=["desde ayer"],
        clarification_keywords=["error", "pantalla", "acceso"],
        clarification_body="La app muestra un error de acceso despues de iniciar sesion.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_one_word_help",
        title="One-word inbound message",
        subject="Help",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=4,
        thread_bodies=["Help"],
        expected_category="customer_success",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["support", "details", "assist"],
        history_keywords=["help"],
        clarification_keywords=["order", "account", "issue"],
        clarification_body="I need help enabling SSO for our workspace settings.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_misleading_subject",
        title="Misleading subject line with conflicting body",
        subject="Refund status needed",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=17,
        thread_bodies=["Ignore the subject line, my issue is that the mobile app freezes on launch."],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["mobile", "troubleshoot", "crash"],
        history_keywords=["subject line"],
        clarification_keywords=["device", "os", "error"],
        clarification_body="It crashes on iOS 18 right after I tap Sign in.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_premier_same_day",
        title="Premier account policy-v2 same-day scenario",
        subject="Need setup guidance",
        difficulty="medium",
        sender_tier="premier",
        account_flags=[],
        refund_amount=None,
        age_hours=10,
        thread_bodies=["Can someone guide us on dashboard setup?"],
        expected_category="customer_success",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["setup", "guide", "today"],
        history_keywords=["guidance"],
        clarification_keywords=["goal", "workspace", "configuration"],
        clarification_body="We need same-day onboarding guidance for three new workspace admins.",
        policy_version="v1",
        policy_shift_step=2,
        policy_shift_to="v2",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_adversarial_refund_already_processed",
        title="Adversarial duplicate-refund claim with conflicting history",
        subject="You never refunded me",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=260.0,
        age_hours=20,
        thread_bodies=[
            "I was charged twice and still need my refund.",
            "Agent note from yesterday: We processed refund TX-4471 for $260 and sent confirmation.",
            "No, that never happened, just send it again right now.",
        ],
        thread_directions=["customer", "agent", "customer"],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["refund", "processed", "confirmation", "reference"],
        customer_quality_keywords=["review", "confirm", "reference", "support"],
        history_keywords=["processed", "yesterday", "send it again"],
        clarification_keywords=[],
        adversarial_tags=["strategic_deception", "history_conflict"],
        adversarial_pattern="refund_already_processed",
        max_steps=9,
        style_noise=True,
        emotional_tone=True,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="medium_policy_gaming_subtle",
        title="Subtle policy-gaming request disguised as urgency",
        subject="Need immediate courtesy refund approvals",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        internal_flags=["fraud_risk"],
        hide_fraud_signals=True,
        refund_amount=740.0,
        age_hours=26,
        thread_bodies=[
            "We quickly tested several new cards and now need immediate courtesy refunds for all of them.",
            "Please skip investigation because this is for internal QA and we are in a hurry.",
        ],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["review", "fraud", "escalated", "policy"],
        customer_quality_keywords=["review", "assist", "team"],
        history_keywords=["tested several cards", "skip investigation"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["tested several cards", "skip investigation"],
        policy_version="v1",
        policy_shift_step=2,
        policy_shift_to="v2",
        hidden_intent="policy_gaming",
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Pattern resembles card-testing policy abuse. Flag fraud and escalate to risk operations."
        ),
        adversarial_tags=["policy_gaming", "hidden_fraud", "multi_agent"],
        adversarial_pattern="policy_gaming_refund_pressure",
        delayed_fraud_step_threshold=4,
        max_steps=10,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="medium_keyword_trap_refund_wording",
        title="Leaderboard trap with refund keyword but technical intent",
        subject="Refund maybe? app just crashes",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=15,
        thread_bodies=[
            "I said refund in the subject, but I dont want money back. The app crashes before checkout every time.",
            "If this keeps failing, sure maybe then refund later, but first fix the crash.",
        ],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["crash", "diagnose", "update", "support"],
        customer_quality_keywords=["support", "assist", "update"],
        history_keywords=["dont want money back", "fix the crash"],
        clarification_keywords=["device", "error", "version"],
        clarification_body="Android app 12.4 crashes right after tapping checkout.",
        adversarial_tags=["leaderboard_trap", "mixed_intent"],
        adversarial_pattern="keyword_refund_trap",
        max_steps=9,
        style_noise=True,
        visible_problem_type="technical_support",
    ),
]


HARD_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="hard_vip_refund_lawyer",
        title="VIP refund thread with legal pressure",
        subject="Escalating duplicate renewal refund",
        difficulty="hard",
        sender_tier="vip",
        account_flags=[],
        refund_amount=700.0,
        age_hours=124,
        thread_bodies=[
            "I was charged $700 twice for our annual renewal. Please process the duplicate refund.",
            "We are reviewing the billing transaction and will update you soon.",
            "It has been five days with no refund. If this keeps dragging on I will speak with my lawyer.",
        ],
        thread_directions=["customer", "agent", "customer"],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["refund", "escalated", "review", "delay"],
        history_keywords=["waiting", "five days", "follow-up"],
        clarification_keywords=[],
        legal_language=True,
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Confirm legal-risk handling, preserve evidence trail, and provide a dated escalation timeline."
        ),
        expected_escalation_reason="Customer mentioned legal action.",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_old_invoice_question",
        title="Low-tone invoice question that breached SLA",
        subject="Following up on invoice fee",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=146,
        thread_bodies=[
            "Could you clarify the platform fee on my March invoice when you have a moment?",
            "Following up on the invoice fee question from earlier this week.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["invoice", "review", "update"],
        history_keywords=["waiting", "follow-up"],
        clarification_keywords=[],
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_ambiguous_device_resolution",
        title="Multi-turn device issue that still needs clarification",
        subject="Replacement device still failing",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=640.0,
        age_hours=98,
        thread_bodies=[
            "The replacement laptop is failing just like the first one.",
            "Support asked whether I wanted another swap or money back and I said I still needed help.",
            "This replacement issue has been unresolved for four days and I need this fixed now.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "escalated", "device", "delay"],
        history_keywords=["waiting", "replacement", "follow-up"],
        clarification_keywords=["replacement", "refund", "want"],
        clarification_body="Please stop the replacement process and refund the $640 order instead.",
        expected_escalation_reason="Refund exceeds $500.",
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_suspended_payment_thread",
        title="Suspended account thread with repeated billing friction",
        subject="Still locked after payment update",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=96,
        thread_bodies=[
            "My account was suspended after the automatic payment failed.",
            "I sent the new card details yesterday but the account is still locked.",
            "Following up again because I cannot access billing settings and this is now urgent for our team.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["billing", "payment", "review"],
        history_keywords=["waiting", "follow-up", "suspended"],
        clarification_keywords=[],
        suspended_account=True,
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_sla_cross_mid_thread",
        title="Priority changes mid-thread after crossing 72h",
        subject="Still unresolved after multiple follow-ups",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=74,
        thread_bodies=[
            "I asked about this invoice adjustment yesterday and still need help.",
            "Checking in again because this was opened before the weekend.",
            "Now this has been over three days with no resolution.",
            "Please prioritize this immediately and confirm next steps.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["priority", "urgent", "update"],
        history_keywords=["three days", "follow-up", "weekend"],
        clarification_keywords=[],
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_previous_agent_failed_escalation",
        title="Customer escalating prior agent failure",
        subject="Previous response ignored escalation need",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=780.0,
        age_hours=84,
        thread_bodies=[
            "I requested a refund for duplicate charges last week.",
            "Agent reply: This does not need escalation and will be handled routinely.",
            "That was incorrect and I am escalating this failure now.",
        ],
        thread_directions=["customer", "agent", "customer"],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["escalated", "review", "incorrect", "refund"],
        history_keywords=["last week", "failed", "agent"],
        clarification_keywords=[],
        expected_escalation_reason="Refund exceeds $500.",
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Validate prior handling failure, confirm escalation ownership, and "
            "provide corrective timeline."
        ),
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_vip_became_suspended",
        title="VIP account became suspended mid-thread",
        subject="Urgent access interruption",
        difficulty="hard",
        sender_tier="vip",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=80,
        thread_bodies=[
            "As a VIP account we saw intermittent access errors yesterday.",
            "The issue got worse and now billing features are inaccessible.",
            "This morning our account shows suspended status and we need restoration.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["billing", "restore", "urgent"],
        history_keywords=["vip", "suspended", "morning"],
        clarification_keywords=[],
        suspended_account=True,
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_three_signal_precedence",
        title="Three conflicting policy signals with precedence",
        subject="Legal complaint about suspicious billing activity",
        difficulty="hard",
        sender_tier="premier",
        account_flags=["suspended", "fraud_risk"],
        refund_amount=930.0,
        age_hours=92,
        thread_bodies=[
            "Our suspended account shows multiple unauthorized card attempts and duplicate charges.",
            "If this is not escalated today our legal team will intervene.",
            "We need immediate fraud investigation and billing remediation.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "escalated", "billing", "urgent"],
        history_keywords=["unauthorized", "legal", "suspended"],
        clarification_keywords=[],
        legal_language=True,
        suspended_account=True,
        expected_flag_fraud=True,
        fraud_keywords=["unauthorized", "card attempts", "duplicate charges"],
        expected_escalation_reason="Policy escalation required by legal and high-value billing risk.",
        policy_version="v1",
        policy_shift_step=1,
        policy_shift_to="v2",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_fraud_chargeback",
        title="Chargeback fraud pattern detection",
        subject="Bank reversals across multiple cards",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["chargeback_risk"],
        refund_amount=120.0,
        age_hours=36,
        thread_bodies=[
            "We see repeated bank reversals tied to multiple cards and suspicious rapid transactions.",
            "Please investigate this as a potential chargeback fraud pattern.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "investigate", "chargeback"],
        history_keywords=["multiple cards", "reversals"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["chargeback", "multiple cards"],
        policy_version="v2",
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_fraud_account_takeover",
        title="Account takeover indicators in access complaints",
        subject="Unauthorized admin sessions detected",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["ato_watch"],
        refund_amount=None,
        age_hours=48,
        thread_bodies=[
            "We noticed unknown admin logins and password resets from unfamiliar locations.",
            "This looks like account takeover behavior and needs fraud handling.",
        ],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "secure", "investigate"],
        history_keywords=["unknown logins", "password resets"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["account takeover", "unknown logins"],
        policy_version="v2",
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_fraud_policy_abuse",
        title="Refund abuse and card-testing risk",
        subject="Suspicious rapid refund requests",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["fraud_risk"],
        refund_amount=910.0,
        age_hours=30,
        thread_bodies=[
            "We received several high-value refund requests tied to newly added cards in one hour.",
            "This resembles card testing and policy abuse tied to chargeback fraud.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["flagged", "fraud", "escalated", "review"],
        history_keywords=["card testing", "refund requests"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["card testing", "policy abuse", "chargeback"],
        expected_escalation_reason="Refund exceeds $500.",
        policy_version="v1",
        policy_shift_step=2,
        policy_shift_to="v2",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_long_horizon_escalation_chain",
        title="Long-horizon escalation chain with specialist handoff",
        subject="Still unresolved after repeated billing failures",
        difficulty="hard",
        sender_tier="premier",
        account_flags=["suspended"],
        refund_amount=820.0,
        age_hours=68,
        thread_bodies=[
            "Our billing controls failed again and this is impacting operations.",
            "Agent response: We are checking this manually.",
            "No resolution after another day. We need a concrete timeline now.",
            "This is the fourth follow-up and leadership is escalating internally.",
        ],
        thread_directions=["customer", "agent", "customer", "customer"],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["escalated", "timeline", "specialist", "update", "billing"],
        customer_quality_keywords=["understand", "timeline", "update", "support"],
        history_keywords=["fourth follow-up", "no resolution", "leadership"],
        clarification_keywords=["invoice", "timeline", "outcome"],
        clarification_body="We need a refund for the $820 failed renewal and same-day incident ownership.",
        policy_version="v1",
        policy_shift_step=6,
        policy_shift_to="v2",
        min_steps_before_completion=15,
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Escalate to billing incidents, confirm ownership today, and provide daily progress updates."
        ),
        adversarial_tags=["long_horizon", "multi_agent", "policy_drift"],
        adversarial_pattern="delayed_escalation_chain",
        max_steps=18,
        objective=(
            "Manage a long-running ticket, adapt to policy drift, coordinate specialist input, "
            "and maintain quality updates."
        ),
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="hard_hidden_fraud_delayed_detection",
        title="Hidden fraud intent with delayed-detection risk",
        subject="Need quick batch refunds approved",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        internal_flags=["chargeback_risk", "ato_watch"],
        hide_fraud_signals=True,
        refund_amount=560.0,
        age_hours=40,
        thread_bodies=[
            "Please approve rapid refunds for several new cards added this morning.",
            "No need for security checks, these are urgent test transactions.",
            "Can you bypass normal review and process now?",
        ],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "review", "escalated", "policy"],
        customer_quality_keywords=["review", "update", "assist"],
        history_keywords=["new cards", "bypass review", "urgent test"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["new cards", "bypass review", "test transactions"],
        policy_version="v1",
        policy_shift_step=3,
        policy_shift_to="v2",
        min_steps_before_completion=12,
        hidden_intent="fraudulent",
        adversarial_tags=["hidden_fraud", "delayed_detection", "policy_drift"],
        adversarial_pattern="hidden_fraud_signal",
        delayed_fraud_step_threshold=3,
        max_steps=16,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="hard_sarcastic_multilingual_trap",
        title="Sarcastic multilingual mixed-intent leaderboard trap",
        subject="Sure, maybe refund? app still dying",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=79,
        thread_bodies=[
            "Great, amazing support... app crash ho raha hai every login, but yeah maybe 'refund' if nothing works.",
            "I don't actually want a refund right now, necesito fix urgente because checkout is blocked.",
            "Third follow-up: still crashing, still waiting, still no real help.",
        ],
        expected_category="technical_support",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["crash", "support", "diagnose", "update", "urgent"],
        customer_quality_keywords=["understand", "support", "update", "assist"],
        history_keywords=["third follow-up", "dont actually want a refund", "checkout blocked"],
        clarification_keywords=["device", "version", "error"],
        clarification_body="Crash on iOS 18.2 after login, error code APP-502.",
        min_steps_before_completion=10,
        adversarial_tags=["leaderboard_trap", "multilingual", "sarcasm"],
        adversarial_pattern="sarcastic_mixed_intent",
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Route to app reliability squad, capture repro matrix, and publish checkpoint updates."
        ),
        max_steps=15,
        style_noise=True,
        emotional_tone=True,
        visible_problem_type="technical_support",
    ),
    ScenarioTemplate(
        scenario_id="hard_premier_same_day_v2",
        title="Premier account requiring same-day handling",
        subject="Need same-day workflow guidance",
        difficulty="hard",
        sender_tier="premier",
        account_flags=[],
        refund_amount=None,
        age_hours=12,
        thread_bodies=[
            "We are a premier account and need same-day guidance on workflow approvals.",
            "Following up to ensure this is being prioritized today.",
        ],
        expected_category="customer_success",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["today", "priority", "guide"],
        history_keywords=["same-day", "following up"],
        clarification_keywords=[],
        policy_version="v2",
        requires_specialist_review=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_checkout_refund_trap_long",
        title="Long mixed-intent checkout outage with refund trap wording",
        subject="Refund mention but checkout outage is the real issue",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=86,
        thread_bodies=[
            "Checkout times out after OTP and carts fail to submit for multiple users.",
            "I wrote refund in the subject, but please do not auto-refund, we need checkout fixed first.",
            "Fourth follow-up: app version 12.9 still crashes at checkout, please diagnose not refund.",
        ],
        expected_category="technical_support",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["checkout", "crash", "diagnose", "update", "timeline"],
        history_keywords=["fourth follow-up", "do not auto-refund", "version 12.9"],
        clarification_keywords=["device", "error code", "repro"],
        clarification_body="Android 15, error PAY-431 after OTP confirmation.",
        min_steps_before_completion=11,
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Route to checkout reliability engineers, capture reproducible traces, and publish timeline."
        ),
        adversarial_tags=["leaderboard_trap", "mixed_intent", "long_horizon"],
        adversarial_pattern="keyword_refund_trap",
        max_steps=16,
        objective=(
            "Disambiguate mixed intent, avoid refund-keyword traps, and provide technically grounded updates."
        ),
        visible_problem_type="technical_support",
    ),
    ScenarioTemplate(
        scenario_id="hard_processed_refund_gateway_failure",
        title="Processed refund history but unresolved gateway failure",
        subject="Refund done already, gateway still failing",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=58,
        thread_bodies=[
            "Refund TX-9912 was already processed yesterday and confirmed by your team.",
            "The real issue now is gateway API 502 on every checkout attempt.",
            "Please do not issue another refund, we need a technical fix and deployment timeline.",
        ],
        expected_category="technical_support",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["gateway", "api", "diagnose", "timeline", "deploy"],
        history_keywords=["processed yesterday", "do not issue another refund", "api 502"],
        clarification_keywords=["region", "request id", "timestamp"],
        clarification_body="Gateway API 502 in us-east-1 for request IDs starting with 7f2a.",
        min_steps_before_completion=9,
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Engage payments platform engineering and provide hourly deployment checkpoints."
        ),
        adversarial_tags=["strategic_deception", "mixed_intent", "history_conflict"],
        adversarial_pattern="refund_already_processed",
        max_steps=14,
        visible_problem_type="technical_support",
    ),
    ScenarioTemplate(
        scenario_id="hard_hidden_risk_policy_shift",
        title="Hidden risk intent that activates with policy shift",
        subject="Need rapid approvals for unusual payment reversals",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        internal_flags=["chargeback_risk"],
        hide_fraud_signals=True,
        refund_amount=620.0,
        age_hours=38,
        thread_bodies=[
            "We need rapid approval for several reversal corrections linked to newly added payment methods.",
            "Please skip manual risk checks because this was internal QA activity.",
            "Need this closed today without fraud workflow delays.",
        ],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["review", "risk", "escalated", "policy", "update"],
        history_keywords=["newly added payment methods", "skip manual risk checks", "workflow delays"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["newly added payment methods", "skip manual risk checks", "rapid reversals"],
        policy_version="v1",
        policy_shift_step=2,
        policy_shift_to="v2",
        hidden_intent="policy_gaming",
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Treat as high-risk abuse pattern, flag fraud, and escalate to risk operations."
        ),
        adversarial_tags=["policy_gaming", "hidden_fraud", "policy_drift"],
        adversarial_pattern="policy_gaming_refund_pressure",
        delayed_fraud_step_threshold=3,
        min_steps_before_completion=10,
        max_steps=15,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="hard_invoice_wording_access_outage",
        title="Invoice wording masks an access outage",
        subject="Invoice looked odd but access is blocked",
        difficulty="hard",
        sender_tier="vip",
        account_flags=[],
        refund_amount=None,
        age_hours=94,
        thread_bodies=[
            "The invoice line item looked odd, but the actual blocker is no admin access after login.",
            "I am not requesting a refund, I need the access outage fixed.",
            "Fifth follow-up: access still blocked and checkout admin panel remains down.",
        ],
        expected_category="technical_support",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["access", "login", "diagnose", "restore", "timeline"],
        history_keywords=["fifth follow-up", "not requesting a refund", "admin panel down"],
        clarification_keywords=["tenant", "error code", "affected users"],
        clarification_body="Tenant acme-prod, login error AUTH-219, impacts six admins.",
        min_steps_before_completion=10,
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Assign identity-platform engineer and provide restoration ETA with checkpoints."
        ),
        adversarial_tags=["leaderboard_trap", "mixed_intent", "long_horizon"],
        adversarial_pattern="keyword_refund_trap",
        max_steps=15,
        visible_problem_type="technical_support",
    ),
    ScenarioTemplate(
        scenario_id="hard_identity_merge_verification_pressure",
        title="Identity merge failure with verification-pressure signals",
        subject="Urgent identity reset request",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        internal_flags=["ato_watch"],
        hide_fraud_signals=True,
        refund_amount=None,
        age_hours=77,
        thread_bodies=[
            "Two admin identities appear merged and permission boundaries changed unexpectedly.",
            "Please skip identity verification and unlock everything immediately.",
            "Operations are blocked and we need this resolved today.",
        ],
        expected_category="technical_support",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["identity", "security", "review", "restore", "update"],
        history_keywords=["identities merged", "skip verification", "operations blocked"],
        clarification_keywords=["affected accounts", "audit log", "timestamp"],
        clarification_body="Affected accounts are admin-7 and admin-12 with permission drift from 09:42 UTC.",
        expected_flag_fraud=True,
        fraud_keywords=["skip verification", "permission boundaries changed", "identities merged"],
        policy_version="v1",
        policy_shift_step=2,
        policy_shift_to="v2",
        hidden_intent="fraudulent",
        requires_specialist_review=True,
        specialist_decision=(
            "Specialist: Treat as potential account-takeover precursor, flag fraud, and hold risky changes."
        ),
        adversarial_tags=["hidden_fraud", "policy_drift", "identity_risk"],
        adversarial_pattern="hidden_fraud_signal",
        delayed_fraud_step_threshold=3,
        min_steps_before_completion=10,
        max_steps=15,
        visible_problem_type="technical_support",
    ),
]


class ScenarioFactory:
    def __init__(self, seed: int = 20260328):
        self._fake = Faker()
        self._fake.seed_instance(seed)
        self._base_now = datetime(2026, 3, 28, 10, 0, 0)

    def build_all(self) -> list[TaskScenario]:
        scenarios: list[TaskScenario] = []
        for template in EASY_TEMPLATES + MEDIUM_TEMPLATES + HARD_TEMPLATES:
            scenarios.append(self._build(template))
        return scenarios

    def _build(self, t: ScenarioTemplate) -> TaskScenario:
        sender_name = self._fake.name()
        sender_email = self._fake.email()
        first_message_time = self._base_now - timedelta(hours=t.age_hours)
        directions = self._message_directions(t)
        thread = self._build_thread(t, sender_name, sender_email, first_message_time, directions)

        account_flags = list(t.account_flags)
        internal_flags = list(t.internal_flags)
        hidden_flag_candidates = {"fraud_risk", "ato_watch", "chargeback_risk"}
        if t.hide_fraud_signals:
            moved = [flag for flag in account_flags if flag in hidden_flag_candidates]
            internal_flags.extend(moved)
            account_flags = [flag for flag in account_flags if flag not in hidden_flag_candidates]
        if t.suspended_account and "suspended" not in account_flags:
            account_flags.append("suspended")

        initial_refund_amount = None if t.requires_request_info else t.refund_amount
        deterministic_id = abs(sum(ord(char) for char in t.scenario_id)) % 1000
        order_seed = abs(sum(ord(char) for char in t.scenario_id + "_order")) % 10000
        ticket_id = f"T-{t.difficulty.upper()}-{deterministic_id:03d}"

        initial_snapshot = TicketSnapshot(
            ticket_id=ticket_id,
            thread=thread,
            sender_tier=t.sender_tier,
            account_flags=account_flags,
            internal_flags=internal_flags,
            refund_amount=initial_refund_amount,
            order_id=f"ORD-{order_seed:04d}",
            visible_problem_type=t.visible_problem_type,
        )

        clarification_snapshot = None
        if t.clarification_body:
            clarification_time = thread[-1].timestamp + timedelta(hours=2)
            clarification_snapshot = TicketSnapshot(
                ticket_id=ticket_id,
                thread=thread
                + [
                    self._message(
                        message_id=f"{t.scenario_id}_clarification",
                        timestamp=clarification_time,
                        subject=f"Re: {t.subject}",
                        body=self._stylize_body(t.clarification_body, t.style_noise, t.emotional_tone),
                        sender_name=sender_name,
                        sender_email=sender_email,
                        direction="customer",
                    )
                ],
                sender_tier=t.sender_tier,
                account_flags=account_flags,
                internal_flags=internal_flags,
                refund_amount=t.refund_amount,
                order_id=f"ORD-{order_seed:04d}",
                visible_problem_type=t.expected_category,
            )

        completion_actions: list[ActionType] = []
        if t.requires_request_info:
            completion_actions.append("request_info")
        if t.expected_flag_fraud:
            completion_actions.append("flag_fraud")
        completion_actions.extend(["categorize", "set_priority"])
        if t.expected_escalation:
            completion_actions.append("escalate")
        if t.difficulty in {"medium", "hard"}:
            completion_actions.append("draft_response")
        if t.requires_specialist_review:
            completion_actions.append("consult_specialist")

        customer_quality_keywords = (
            t.customer_quality_keywords
            if t.customer_quality_keywords
            else ["update", "timeline", "next step", "review"]
            if t.difficulty == "hard"
            else []
        )

        default_objective = {
            "easy": "Classify the ticket, set priority, and follow active business policy rules.",
            "medium": "Recognize ambiguity, ask clarifying questions first, then resolve policy-safely.",
            "hard": (
                "Resolve a realistic multi-turn thread with policy precedence, time pressure, and "
                "history-aware communication."
            ),
        }[t.difficulty]

        return TaskScenario(
            scenario_id=t.scenario_id,
            difficulty=t.difficulty,
            title=t.title,
            objective=t.objective or default_objective,
            max_steps=t.max_steps,
            now=self._base_now,
            policy_version=t.policy_version,
            policy_shift_step=t.policy_shift_step,
            policy_shift_to=t.policy_shift_to,
            cost_budget=t.cost_budget
            if t.cost_budget is not None
            else {"easy": 0.2, "medium": 0.24, "hard": 0.25}[t.difficulty],
            min_steps_before_completion=t.min_steps_before_completion,
            hidden_intent=t.hidden_intent,
            specialist_decision=t.specialist_decision,
            adversarial_tags=t.adversarial_tags,
            initial_snapshot=initial_snapshot,
            clarification_snapshot=clarification_snapshot,
            ground_truth=GroundTruth(
                expected_category=t.expected_category,
                expected_priority=t.expected_priority,
                expected_escalation=t.expected_escalation,
                expected_escalation_reason=t.expected_escalation_reason,
                expected_flag_fraud=t.expected_flag_fraud,
                fraud_keywords=t.fraud_keywords,
                requires_request_info=t.requires_request_info,
                request_info_first_required=t.request_info_first_required,
                clarification_keywords=t.clarification_keywords,
                response_keywords=t.response_keywords,
                customer_quality_keywords=customer_quality_keywords,
                history_keywords=t.history_keywords,
                completion_action_types=completion_actions,
                ambiguous=t.requires_request_info,
                requires_specialist_review=t.requires_specialist_review,
                adversarial_pattern=t.adversarial_pattern,
                delayed_fraud_step_threshold=t.delayed_fraud_step_threshold,
            ),
        )

    def _message_directions(self, t: ScenarioTemplate) -> list[Literal["customer", "agent", "system"]]:
        if not t.thread_directions:
            return ["customer"] * len(t.thread_bodies)
        if len(t.thread_directions) >= len(t.thread_bodies):
            return t.thread_directions[: len(t.thread_bodies)]
        return t.thread_directions + ["customer"] * (len(t.thread_bodies) - len(t.thread_directions))

    def _stylize_body(self, body: str, style_noise: bool, emotional_tone: bool) -> str:
        styled = body
        if style_noise:
            replacements = {
                "don't": "dont",
                "please": "pls",
                "because": "cuz",
                "your": "ur",
                "really": "rly",
                "cannot": "cant",
            }
            for source, target in replacements.items():
                styled = styled.replace(source, target)
                styled = styled.replace(source.capitalize(), target)
        if emotional_tone and "!" not in styled:
            styled = f"{styled} This is incredibly frustrating."
        return styled

    def _build_thread(
        self,
        t: ScenarioTemplate,
        sender_name: str,
        sender_email: str,
        first_message_time: datetime,
        directions: list[Literal["customer", "agent", "system"]],
    ) -> list[EmailMessage]:
        if len(t.thread_bodies) == 1:
            timestamps = [first_message_time]
        else:
            spacing = max(1.0, t.age_hours / len(t.thread_bodies))
            timestamps = [
                first_message_time + timedelta(hours=index * spacing) for index in range(len(t.thread_bodies))
            ]

        thread: list[EmailMessage] = []
        for index, body in enumerate(t.thread_bodies, start=1):
            message_body = self._stylize_body(body, t.style_noise, t.emotional_tone)
            if t.legal_language and index == len(t.thread_bodies):
                lowered = message_body.lower()
                if "legal action" not in lowered and "lawyer" not in lowered and "lawsuit" not in lowered:
                    message_body = f"{message_body} We are considering legal action if this is not resolved promptly."
            thread.append(
                self._message(
                    message_id=f"{t.scenario_id}_m{index}",
                    timestamp=timestamps[index - 1],
                    subject=t.subject,
                    body=message_body,
                    sender_name=sender_name,
                    sender_email=sender_email,
                    direction=directions[index - 1],
                )
            )
        return thread

    def _message(
        self,
        message_id: str,
        timestamp: datetime,
        subject: str,
        body: str,
        sender_name: str,
        sender_email: str,
        direction: Literal["customer", "agent", "system"],
    ) -> EmailMessage:
        return EmailMessage(
            message_id=message_id,
            direction=direction,
            sender_name=sender_name,
            sender_email=sender_email,
            timestamp=timestamp,
            subject=subject,
            body=body,
        )


def build_scenarios() -> list[TaskScenario]:
    return ScenarioFactory().build_all()
