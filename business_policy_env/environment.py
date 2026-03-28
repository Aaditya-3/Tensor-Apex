from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from pydantic import ValidationError

from .models import Action, ActionRecord, EpisodePhase, Observation, TaskScenario, TicketSnapshot
from .policies import check_policy_violations, policy_rules_for
from .rewards import current_progress, invalid_action_breakdown, shaped_reward
from .tasks import build_ground_truth_payload, compute_issue_age_hours, scenario_registry, scenarios_for_task


class BusinessPolicyComplianceEnv:
    def __init__(self) -> None:
        self._scenario_registry = scenario_registry()
        self._task_cursors: dict[str, int] = defaultdict(int)
        self._connection = self._create_connection()
        self.current_scenario: TaskScenario | None = None
        self.action_history: list[ActionRecord] = []
        self.clarification_received = False
        self.episode_phase = EpisodePhase.initial
        self._simulated_offset_hours = 0.0
        self._snooze_crossed_sla = False
        self.done = False

    def _create_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episode_actions (
                step_index INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                valid INTEGER NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    def _reset_connection(self) -> None:
        self._connection.close()
        self._connection = self._create_connection()

    def available_tasks(self) -> dict[str, list[str]]:
        return {
            "easy": [scenario.scenario_id for scenario in scenarios_for_task("easy")],
            "medium": [scenario.scenario_id for scenario in scenarios_for_task("medium")],
            "hard": [scenario.scenario_id for scenario in scenarios_for_task("hard")],
        }

    def _select_scenario(self, task_name: str | None, scenario_id: str | None) -> TaskScenario:
        if scenario_id:
            return self._scenario_registry[scenario_id]

        selected_task = task_name or "easy"
        candidates = scenarios_for_task(selected_task)
        cursor = self._task_cursors[selected_task] % len(candidates)
        self._task_cursors[selected_task] += 1
        return candidates[cursor]

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

    def _step_timestamp(self, step_index: int) -> datetime:
        scenario = self._active_snapshot()
        return scenario.now + timedelta(seconds=step_index)

    def _log_action(self, record: ActionRecord) -> None:
        self._connection.execute(
            "INSERT INTO episode_actions(step_index, action_type, payload, timestamp, valid) VALUES (?, ?, ?, ?, ?)",
            (
                record.step_index,
                record.action.action_type,
                json.dumps(record.action.model_dump(mode="json")),
                record.timestamp.isoformat(),
                int(record.valid),
            ),
        )
        self._connection.commit()

    def _episode_log(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT step_index, action_type, payload, timestamp, valid FROM episode_actions ORDER BY step_index"
        ).fetchall()
        return [dict(row) for row in rows]

    def _observation(self) -> Observation:
        scenario = self._active_snapshot()
        snapshot = self._current_snapshot()
        return Observation(
            scenario_id=scenario.scenario_id,
            difficulty=scenario.difficulty,
            current_email=snapshot.thread[-1],
            thread=snapshot.thread,
            sender_tier=snapshot.sender_tier,
            account_flags=snapshot.account_flags,
            refund_amount=snapshot.refund_amount,
            issue_age_hours=self._issue_age_hours(),
            emails_remaining=1,
            steps_taken=len(self.action_history),
            max_steps=scenario.max_steps,
            action_history=self.action_history,
            policy_rules=policy_rules_for(scenario.policy_version),
            policy_version=scenario.policy_version,
            task_objective=scenario.objective,
            clarification_received=self.clarification_received,
            episode_phase=self.episode_phase,
        )

    def _completion_reached(self) -> bool:
        scenario = self._active_snapshot()
        completed_types = {record.action.action_type for record in self.action_history}
        required_types = set(scenario.ground_truth.completion_action_types)
        return required_types.issubset(completed_types)

    def _advance_phase(self, action: Action) -> None:
        phase = self.episode_phase
        resolving_actions = {"categorize", "set_priority", "escalate", "flag_fraud", "draft_response", "mark_spam"}

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
        self.done = False
        self._reset_connection()
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
            scenario.policy_version,
            prior_actions=prior_actions,
        )

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
        self._log_action(record)

        if (
            action.action_type == "request_info"
            and scenario.clarification_snapshot is not None
            and not self.clarification_received
        ):
            self.clarification_received = True

        if len(self.action_history) >= scenario.max_steps or self._completion_reached():
            self.done = True

        self._advance_phase(action)

        grading_payload = build_ground_truth_payload(scenario, self._grade_snapshot())
        actions = [item.action for item in self.action_history]
        reward_breakdown = shaped_reward(
            actions,
            grading_payload,
            self.done,
            scenario.max_steps,
            policy_violations,
            snooze_crossed_sla=self._snooze_crossed_sla,
            fraud_expected=scenario.ground_truth.expected_flag_fraud,
        )
        progress_score, components = current_progress(actions, grading_payload)
        observation = self._observation()
        info_step: dict[str, Any] = {
            "valid_action": True,
            "final_score": progress_score if self.done else None,
            "partial_score": None if self.done else progress_score,
            "policy_violations": policy_violations,
            "reward_breakdown": reward_breakdown.components,
            "component_scores": components,
            "explanation": reward_breakdown.explanation,
        }
        return observation, reward_breakdown.reward, self.done, info_step

    def state(self) -> dict[str, Any]:
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
        return {
            "active": True,
            "ground_truth": build_ground_truth_payload(scenario, active_snapshot),
            "dataset_reference": scenario.model_dump(mode="json"),
            "episode_log": self._episode_log(),
            "current_task_configuration": {
                "difficulty": scenario.difficulty,
                "max_steps": scenario.max_steps,
                "objective": scenario.objective,
                "title": scenario.title,
                "policy_version": scenario.policy_version,
            },
            "policy_rules": policy_rules_for(scenario.policy_version),
            "internal_variables": {
                "clarification_received": self.clarification_received,
                "episode_phase": self.episode_phase,
                "simulated_offset_hours": self._simulated_offset_hours,
                "snooze_crossed_sla": self._snooze_crossed_sla,
                "done": self.done,
                "steps_taken": len(self.action_history),
            },
        }
