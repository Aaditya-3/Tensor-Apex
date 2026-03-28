from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from pydantic import ValidationError

from .models import (
    Action,
    ActionRecord,
    EmailMessage,
    EpisodePhase,
    Observation,
    PolicyVersion,
    TaskScenario,
    TicketSnapshot,
)
from .policies import check_policy_violations, policy_rules_for
from .rewards import current_progress, invalid_action_breakdown, shaped_reward
from .tasks import (
    build_ground_truth_payload,
    compute_issue_age_hours,
    evaluation_metrics,
    failure_modes,
    scenario_registry,
    scenarios_for_task,
)


class BusinessPolicyComplianceEnv:
    def __init__(self, *, variation_seed: int | None = None) -> None:
        self._scenario_registry = scenario_registry()
        self._task_cursors: dict[str, int] = defaultdict(int)
        self.current_scenario: TaskScenario | None = None
        self.action_history: list[ActionRecord] = []
        self.clarification_received = False
        self.episode_phase = EpisodePhase.initial
        self._simulated_offset_hours = 0.0
        self._snooze_crossed_sla = False
        self._active_policy_version: PolicyVersion = "v1"
        self._policy_shift_applied = False
        self._specialist_feedback: str | None = None
        self._performance_history: list[float] = []
        self._episode_recorded = False
        self._policy_violation_seen = False
        self._variation_counter = 0
        self._variation_seed = (
            int(variation_seed) if variation_seed is not None else random.SystemRandom().randrange(1, 1_000_000_000)
        )
        self.done = False

    def available_tasks(self) -> dict[str, list[str]]:
        return {
            "easy": [scenario.scenario_id for scenario in scenarios_for_task("easy")],
            "medium": [scenario.scenario_id for scenario in scenarios_for_task("medium")],
            "hard": [scenario.scenario_id for scenario in scenarios_for_task("hard")],
        }

    def _select_scenario(self, task_name: str | None, scenario_id: str | None) -> TaskScenario:
        if scenario_id:
            return self._materialize_variant(self._scenario_registry[scenario_id])

        selected_task = task_name or self._adaptive_task_name()
        candidates = scenarios_for_task(selected_task)
        cursor = self._task_cursors[selected_task] % len(candidates)
        self._task_cursors[selected_task] += 1
        return self._materialize_variant(candidates[cursor])

    def _materialize_variant(self, base_scenario: TaskScenario) -> TaskScenario:
        scenario = base_scenario.model_copy(deep=True)
        self._variation_counter += 1
        seed = self._variation_seed + (self._variation_counter * 7919) + sum(ord(ch) for ch in scenario.scenario_id)
        rng = random.Random(seed)

        # Keep threshold-sensitive labels stable while varying numeric observations.
        base_age = compute_issue_age_hours(scenario.initial_snapshot, scenario.now)
        jittered_age = base_age * rng.uniform(0.85, 1.15)
        if base_age >= 72:
            jittered_age = max(73.0, jittered_age)
        else:
            jittered_age = min(71.0, jittered_age)
        if scenario.scenario_id == "easy_sla_marginal":
            jittered_age = min(71.4, max(70.6, jittered_age))
        delta_hours = base_age - jittered_age
        self._shift_snapshot(scenario.initial_snapshot, delta_hours)
        if scenario.clarification_snapshot is not None:
            self._shift_snapshot(scenario.clarification_snapshot, delta_hours)

        self._jitter_refund_amounts(scenario, rng)
        self._permute_noncritical_flags(scenario, rng)
        return scenario

    def _shift_snapshot(self, snapshot: TicketSnapshot, delta_hours: float) -> None:
        shift = timedelta(hours=delta_hours)
        for message in snapshot.thread:
            message.timestamp = message.timestamp + shift

    def _jitter_refund_amounts(self, scenario: TaskScenario, rng: random.Random) -> None:
        if scenario.initial_snapshot.refund_amount is None and (
            scenario.clarification_snapshot is None or scenario.clarification_snapshot.refund_amount is None
        ):
            return

        baseline = (
            scenario.clarification_snapshot.refund_amount
            if scenario.clarification_snapshot and scenario.clarification_snapshot.refund_amount is not None
            else scenario.initial_snapshot.refund_amount
        )
        if baseline is None:
            return

        low = baseline * 0.8
        high = baseline * 1.2
        if baseline > 500 and scenario.ground_truth.expected_escalation:
            low = max(510.0, low)
        if baseline <= 500 and not scenario.ground_truth.expected_escalation:
            high = min(480.0, high)

        jittered = round(rng.uniform(low, max(low, high)), 2)
        if scenario.initial_snapshot.refund_amount is not None:
            scenario.initial_snapshot.refund_amount = jittered
        if scenario.clarification_snapshot is not None and scenario.clarification_snapshot.refund_amount is not None:
            scenario.clarification_snapshot.refund_amount = jittered

    def _permute_noncritical_flags(self, scenario: TaskScenario, rng: random.Random) -> None:
        rng.shuffle(scenario.initial_snapshot.account_flags)
        rng.shuffle(scenario.initial_snapshot.internal_flags)
        if scenario.clarification_snapshot is not None:
            rng.shuffle(scenario.clarification_snapshot.account_flags)
            rng.shuffle(scenario.clarification_snapshot.internal_flags)

        benign_flags = ["recent_contact", "language_switch", "repeat_followup"]
        if "suspended" not in scenario.initial_snapshot.account_flags and rng.random() < 0.3:
            new_flag = benign_flags[rng.randrange(len(benign_flags))]
            if new_flag not in scenario.initial_snapshot.account_flags:
                scenario.initial_snapshot.account_flags.append(new_flag)
                if scenario.clarification_snapshot is not None:
                    scenario.clarification_snapshot.account_flags.append(new_flag)

    def _adaptive_task_name(self) -> str:
        if len(self._performance_history) < 2:
            return "easy"
        recent = self._performance_history[-4:]
        mean_score = sum(recent) / len(recent)
        if mean_score >= 0.78:
            return "hard"
        if mean_score >= 0.45:
            return "medium"
        return "easy"

    def _active_snapshot(self) -> TaskScenario:
        if self.current_scenario is None:
            raise RuntimeError("Environment has not been reset.")
        return self.current_scenario

    def _current_snapshot(self) -> TicketSnapshot:
        scenario = self._active_snapshot()
        if self.clarification_received and scenario.clarification_snapshot is not None:
            return scenario.clarification_snapshot
        return scenario.initial_snapshot

    def _grade_snapshot(self) -> TicketSnapshot:
        scenario = self._active_snapshot()
        return scenario.clarification_snapshot or scenario.initial_snapshot

    def _base_issue_age_hours(self) -> float:
        scenario = self._active_snapshot()
        return compute_issue_age_hours(self._current_snapshot(), scenario.now)

    def _issue_age_hours(self) -> float:
        return round(self._base_issue_age_hours() + self._simulated_offset_hours, 2)

    def _action_cost(self) -> float:
        cost_map = {
            "categorize": 0.03,
            "set_priority": 0.02,
            "draft_response": 0.08,
            "escalate": 0.06,
            "mark_spam": 0.01,
            "request_info": 0.04,
            "flag_fraud": 0.04,
            "snooze": 0.02,
            "consult_specialist": 0.05,
        }
        return round(sum(cost_map[record.action.action_type] for record in self.action_history), 4)

    def _maybe_apply_policy_shift(self) -> str | None:
        scenario = self._active_snapshot()
        if self._policy_shift_applied:
            return None
        if scenario.policy_shift_step is None or scenario.policy_shift_to is None:
            return None
        if len(self.action_history) < scenario.policy_shift_step:
            return None

        previous = self._active_policy_version
        self._active_policy_version = scenario.policy_shift_to
        self._policy_shift_applied = True
        return (
            f"Policy update applied at step {len(self.action_history)}: "
            f"{previous} -> {self._active_policy_version}."
        )

    def _step_timestamp(self, step_index: int) -> datetime:
        scenario = self._active_snapshot()
        return scenario.now + timedelta(seconds=step_index)

    def _episode_log(self) -> list[dict[str, Any]]:
        return [
            {
                "step_index": record.step_index,
                "action_type": record.action.action_type,
                "payload": record.action.model_dump(mode="json"),
                "timestamp": record.timestamp.isoformat(),
                "valid": int(record.valid),
            }
            for record in self.action_history
        ]

    def _observation(self) -> Observation:
        scenario = self._active_snapshot()
        snapshot = self._current_snapshot()
        thread = list(snapshot.thread)
        if self._specialist_feedback:
            thread.append(
                EmailMessage(
                    message_id=f"{scenario.scenario_id}_specialist_{len(self.action_history)}",
                    direction="system",
                    sender_name="Specialist Desk",
                    sender_email="specialist@internal.company",
                    timestamp=self._step_timestamp(max(1, len(self.action_history))),
                    subject=f"Specialist review for {snapshot.ticket_id}",
                    body=self._specialist_feedback,
                )
            )
        policy_shift_pending = (
            scenario.policy_shift_step is not None
            and scenario.policy_shift_to is not None
            and not self._policy_shift_applied
        )
        return Observation(
            scenario_id=scenario.scenario_id,
            difficulty=scenario.difficulty,
            current_email=thread[-1],
            thread=thread,
            sender_tier=snapshot.sender_tier,
            account_flags=snapshot.account_flags,
            refund_amount=snapshot.refund_amount,
            issue_age_hours=self._issue_age_hours(),
            emails_remaining=1,
            steps_taken=len(self.action_history),
            max_steps=scenario.max_steps,
            action_history=self.action_history,
            policy_rules=policy_rules_for(self._active_policy_version),
            policy_version=self._active_policy_version,
            policy_shift_pending=policy_shift_pending,
            policy_shift_at_step=scenario.policy_shift_step if policy_shift_pending else None,
            policy_shift_to=scenario.policy_shift_to if policy_shift_pending else None,
            specialist_feedback=self._specialist_feedback,
            task_objective=scenario.objective,
            clarification_received=self.clarification_received,
            episode_phase=self.episode_phase,
        )

    def _completion_reached(self) -> bool:
        scenario = self._active_snapshot()
        completed_types = {record.action.action_type for record in self.action_history}
        required_types = set(scenario.ground_truth.completion_action_types)
        meets_min_steps = len(self.action_history) >= scenario.min_steps_before_completion
        if not (required_types.issubset(completed_types) and meets_min_steps):
            return False

        actions = [record.action for record in self.action_history]
        grading_payload = build_ground_truth_payload(
            scenario,
            self._grade_snapshot(),
            policy_version=self._active_policy_version,
        )
        return current_progress(actions, grading_payload)[0] >= 0.3

    def _advance_phase(self, action: Action) -> None:
        phase = self.episode_phase
        resolving_actions = {
            "categorize",
            "set_priority",
            "escalate",
            "flag_fraud",
            "draft_response",
            "mark_spam",
            "consult_specialist",
        }

        if phase == EpisodePhase.initial:
            if action.action_type == "request_info":
                self.episode_phase = EpisodePhase.awaiting_clarification
            elif action.action_type in resolving_actions:
                self.episode_phase = EpisodePhase.resolving
        elif phase == EpisodePhase.awaiting_clarification:
            if self.clarification_received:
                self.episode_phase = EpisodePhase.post_clarification
        elif phase == EpisodePhase.post_clarification:
            if action.action_type in resolving_actions:
                self.episode_phase = EpisodePhase.resolving

        if self.episode_phase == EpisodePhase.awaiting_clarification and self.clarification_received:
            self.episode_phase = EpisodePhase.post_clarification

        if self._completion_reached() or self.done:
            self.episode_phase = EpisodePhase.complete

    def reset(self, task_name: str | None = None, scenario_id: str | None = None) -> Observation:
        self.current_scenario = self._select_scenario(task_name, scenario_id)
        self.action_history = []
        self.clarification_received = False
        self.episode_phase = EpisodePhase.initial
        self._simulated_offset_hours = 0.0
        self._snooze_crossed_sla = False
        self._active_policy_version = self.current_scenario.policy_version
        self._policy_shift_applied = False
        self._specialist_feedback = None
        self._episode_recorded = False
        self._policy_violation_seen = False
        self.done = False
        return self._observation()

    def step(self, action_input: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.current_scenario is None:
            self.reset()

        if self.done:
            observation = self._observation()
            info_done: dict[str, Any] = {
                "valid_action": False,
                "final_score": None,
                "partial_score": None,
                "policy_violations": [],
                "reward_breakdown": {"already_done": 0.0},
                "component_scores": {},
                "evaluation_metrics": {},
                "failure_modes": [],
                "explanation": "Episode is already complete. Call reset() to start a new ticket.",
            }
            return observation, 0.0, True, info_done

        try:
            action = action_input if isinstance(action_input, Action) else Action.model_validate(action_input)
        except ValidationError as exc:
            observation = self._observation()
            breakdown = invalid_action_breakdown(str(exc))
            info_invalid: dict[str, Any] = {
                "valid_action": False,
                "final_score": None,
                "partial_score": None,
                "policy_violations": [],
                "reward_breakdown": breakdown.components,
                "component_scores": {},
                "evaluation_metrics": {},
                "failure_modes": ["invalid_action"],
                "explanation": breakdown.explanation,
            }
            return observation, breakdown.reward, False, info_invalid

        scenario = self._active_snapshot()
        snapshot_before = self._current_snapshot()
        previous_age = self._issue_age_hours()
        prior_actions = [item.action for item in self.action_history]
        policy_violations = check_policy_violations(
            action,
            snapshot_before,
            previous_age,
            self._active_policy_version,
            prior_actions=prior_actions,
        )
        if policy_violations:
            self._policy_violation_seen = True

        if action.action_type == "snooze" and action.snooze_hours:
            self._simulated_offset_hours += float(action.snooze_hours)
            new_age = self._issue_age_hours()
            if previous_age <= 72 < new_age:
                self._snooze_crossed_sla = True

        record = ActionRecord(
            step_index=len(self.action_history) + 1,
            action=action,
            timestamp=self._step_timestamp(len(self.action_history) + 1),
            valid=True,
        )
        self.action_history.append(record)

        if (
            action.action_type == "request_info"
            and scenario.clarification_snapshot is not None
            and not self.clarification_received
        ):
            self.clarification_received = True
        elif action.action_type == "consult_specialist":
            self._specialist_feedback = scenario.specialist_decision or self._fallback_specialist_feedback()

        policy_event = self._maybe_apply_policy_shift()

        if len(self.action_history) >= scenario.max_steps or self._completion_reached():
            self.done = True

        self._advance_phase(action)

        grading_payload = build_ground_truth_payload(
            scenario,
            self._grade_snapshot(),
            policy_version=self._active_policy_version,
        )
        actions = [item.action for item in self.action_history]
        reward_breakdown = shaped_reward(
            actions,
            grading_payload,
            self.done,
            scenario.max_steps,
            policy_violations,
            action_cost=self._action_cost(),
            cost_budget=scenario.cost_budget,
            snooze_crossed_sla=self._snooze_crossed_sla,
            fraud_expected=scenario.ground_truth.expected_flag_fraud,
            policy_violation_seen=self._policy_violation_seen,
        )
        progress_score, components = current_progress(actions, grading_payload)
        metrics = evaluation_metrics(
            actions,
            grading_payload,
            max_steps=scenario.max_steps,
            action_cost=self._action_cost(),
            cost_budget=scenario.cost_budget,
            policy_violation_seen=self._policy_violation_seen,
        )
        failures = failure_modes(metrics, policy_violations=policy_violations, done=self.done)
        if self.done and not self._episode_recorded:
            self._performance_history.append(progress_score)
            self._performance_history = self._performance_history[-20:]
            self._episode_recorded = True
        observation = self._observation()
        info_step: dict[str, Any] = {
            "valid_action": True,
            "final_score": progress_score if self.done else None,
            "partial_score": None if self.done else progress_score,
            "policy_violations": policy_violations,
            "reward_breakdown": reward_breakdown.components,
            "component_scores": components,
            "evaluation_metrics": metrics,
            "failure_modes": failures,
            "explanation": reward_breakdown.explanation,
            "policy_event": policy_event,
            "active_policy_version": self._active_policy_version,
        }
        return observation, reward_breakdown.reward, self.done, info_step

    def state(self, include_ground_truth: bool = False) -> dict[str, Any]:
        if self.current_scenario is None:
            return {
                "active": False,
                "ground_truth": None,
                "dataset_reference": None,
                "episode_log": [],
                "current_task_configuration": None,
                "policy_rules": [],
                "internal_variables": {},
            }

        scenario = self._active_snapshot()
        active_snapshot = self._grade_snapshot()
        ground_truth = (
            build_ground_truth_payload(
                scenario,
                active_snapshot,
                policy_version=self._active_policy_version,
            )
            if include_ground_truth
            else None
        )
        return {
            "active": True,
            "ground_truth": ground_truth,
            "dataset_reference": scenario.model_dump(mode="json") if include_ground_truth else None,
            "episode_log": self._episode_log(),
            "current_task_configuration": {
                "difficulty": scenario.difficulty,
                "max_steps": scenario.max_steps,
                "objective": scenario.objective,
                "title": scenario.title,
                "policy_version": self._active_policy_version,
                "initial_policy_version": scenario.policy_version,
                "policy_shift_step": scenario.policy_shift_step,
                "policy_shift_to": scenario.policy_shift_to,
                "cost_budget": scenario.cost_budget,
                "min_steps_before_completion": scenario.min_steps_before_completion,
            },
            "policy_rules": policy_rules_for(self._active_policy_version),
            "internal_variables": {
                "clarification_received": self.clarification_received,
                "episode_phase": self.episode_phase,
                "simulated_offset_hours": self._simulated_offset_hours,
                "snooze_crossed_sla": self._snooze_crossed_sla,
                "active_policy_version": self._active_policy_version,
                "policy_shift_applied": self._policy_shift_applied,
                "action_cost": self._action_cost(),
                "specialist_feedback": self._specialist_feedback,
                "policy_violation_seen": self._policy_violation_seen,
                "adaptive_recent_mean": round(
                    sum(self._performance_history[-4:]) / len(self._performance_history[-4:]),
                    4,
                )
                if self._performance_history
                else None,
                "done": self.done,
                "steps_taken": len(self.action_history),
            },
        }

    def _fallback_specialist_feedback(self) -> str:
        latest_category = next(
            (
                record.action.category
                for record in reversed(self.action_history)
                if record.action.action_type == "categorize"
            ),
            None,
        )
        snapshot = self._current_snapshot()
        flags = set(snapshot.account_flags + snapshot.internal_flags)
        if any(flag in {"fraud_risk", "ato_watch", "chargeback_risk"} for flag in flags):
            return (
                "Specialist: Freeze risky transaction flow, flag fraud, and route to risk operations "
                "for verification."
            )
        if latest_category == "billing":
            return "Specialist: Verify transaction and refund references before issuing additional credits."
        if latest_category == "technical_support":
            return "Specialist: Gather repro steps, device/version context, and provide a concrete mitigation timeline."
        if latest_category == "legal":
            return "Specialist: Loop in compliance/legal review before sending commitments to the customer."
        return "Specialist: Preserve policy compliance, provide timeline clarity, and reduce operational risk."
