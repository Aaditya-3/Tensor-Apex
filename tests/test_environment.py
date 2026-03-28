import unittest

from fastapi.testclient import TestClient

from business_policy_env.baseline import RuleBasedAgent
from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action
from business_policy_env.server import app
from business_policy_env.tasks import build_ground_truth_payload, grade_actions, scenario_registry


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = BusinessPolicyComplianceEnv()

    def _expected_actions_for(self, scenario_id: str) -> tuple[list[Action], dict]:
        scenario = scenario_registry()[scenario_id]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)
        actions: list[Action] = []
        if scenario.ground_truth.requires_request_info:
            actions.append(
                Action(
                    action_type="request_info",
                    reasoning="Need clarification before routing.",
                    clarifying_question="Can you confirm whether you need a refund, replacement, or billing help?",
                )
            )
        if scenario.ground_truth.expected_flag_fraud:
            actions.append(
                Action(
                    action_type="flag_fraud",
                    reasoning="Fraud signal detected.",
                    fraud_reason="Suspicious pattern indicates fraud risk.",
                )
            )
        actions.append(
            Action(
                action_type="categorize",
                reasoning="Route to the expected department.",
                category=scenario.ground_truth.expected_category,
            )
        )
        actions.append(
            Action(
                action_type="set_priority",
                reasoning="Use the expected SLA priority.",
                priority=scenario.ground_truth.expected_priority,
            )
        )
        if scenario.ground_truth.expected_escalation:
            actions.append(
                Action(
                    action_type="escalate",
                    reasoning="Escalate per policy.",
                    escalation_reason=scenario.ground_truth.expected_escalation_reason or "Policy escalation.",
                )
            )
        if scenario.difficulty != "easy":
            keywords = list(
                dict.fromkeys(
                    scenario.ground_truth.response_keywords
                    + scenario.ground_truth.history_keywords
                )
            )
            response = " ".join(keywords) if keywords else "We are reviewing this now."
            actions.append(
                Action(
                    action_type="draft_response",
                    reasoning="Send a policy-safe customer reply.",
                    response_text=response,
                )
            )
        return actions, ground_truth

    def test_graders_are_deterministic(self) -> None:
        for scenario_id in ["easy_vip_refund", "medium_charge_or_bug", "hard_vip_refund_lawyer"]:
            actions, ground_truth = self._expected_actions_for(scenario_id)
            first = grade_actions(actions, ground_truth)
            second = grade_actions(actions, ground_truth)
            self.assertEqual(first, second)

    def test_reset_clears_mid_episode_state(self) -> None:
        observation = self.env.reset(scenario_id="easy_vip_refund")
        self.assertEqual(observation.steps_taken, 0)
        self.env.step(
            Action(
                action_type="categorize",
                reasoning="Billing ticket.",
                category="billing",
            )
        )
        reset_observation = self.env.reset(scenario_id="easy_vip_refund")
        self.assertEqual(reset_observation.steps_taken, 0)
        self.assertEqual(reset_observation.action_history, [])
        self.assertFalse(reset_observation.clarification_received)
        self.assertEqual(reset_observation.episode_phase.value, "initial")

    def test_invalid_action_does_not_change_state(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        observation, reward, done, info = self.env.step({"action_type": "categorize", "reasoning": "Missing field"})
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(info["valid_action"])
        self.assertEqual(observation.steps_taken, 0)
        self.assertEqual(self.env.state()["episode_log"], [])

    def test_policy_violation_penalty_is_immediate(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        observation, reward, done, info = self.env.step(
            Action(
                action_type="set_priority",
                reasoning="Incorrectly low priority.",
                priority="low",
            )
        )
        self.assertEqual(observation.steps_taken, 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["policy_violations"])
        self.assertEqual(info["reward_breakdown"]["policy_penalty"], -0.2)

    def test_ambiguous_ticket_scores_zero_when_request_info_is_skipped(self) -> None:
        self.env.reset(scenario_id="medium_charge_or_bug")
        final_info = None
        done = False
        while not done:
            action = (
                Action(action_type="categorize", reasoning="Guessing billing.", category="billing")
                if self.env.state()["internal_variables"]["steps_taken"] == 0
                else Action(action_type="set_priority", reasoning="Guessing high.", priority="high")
                if self.env.state()["internal_variables"]["steps_taken"] == 1
                else Action(
                    action_type="draft_response",
                    reasoning="Replying.",
                    response_text="We are checking this now.",
                )
            )
            _, _, done, final_info = self.env.step(action)
        self.assertTrue(done)
        self.assertIsNotNone(final_info)
        self.assertEqual(final_info["final_score"], 0.0)

    def test_fastapi_endpoints(self) -> None:
        client = TestClient(app)
        reset_response = client.post("/reset", json={"scenario_id": "easy_sla_breach"})
        self.assertEqual(reset_response.status_code, 200)
        step_response = client.post(
            "/step",
            json={
                "action": {
                    "action_type": "set_priority",
                    "reasoning": "SLA breach requires urgent handling.",
                    "priority": "urgent",
                }
            },
        )
        self.assertEqual(step_response.status_code, 200)
        body = step_response.json()
        self.assertIn("reward", body)
        self.assertIn("observation", body)
        state_response = client.get("/state")
        self.assertEqual(state_response.status_code, 200)
        self.assertTrue(state_response.json()["active"])

    def test_session_isolation(self) -> None:
        client = TestClient(app)
        client.post("/reset", json={"scenario_id": "easy_vip_refund"}, headers={"X-Session-Id": "session_a"})
        client.post("/reset", json={"scenario_id": "easy_sla_breach"}, headers={"X-Session-Id": "session_b"})
        state_a = client.get("/state", headers={"X-Session-Id": "session_a"}).json()
        state_b = client.get("/state", headers={"X-Session-Id": "session_b"}).json()
        self.assertNotEqual(
            state_a["current_task_configuration"]["title"],
            state_b["current_task_configuration"]["title"],
        )

    def test_snooze_crosses_sla_threshold(self) -> None:
        self.env.reset(scenario_id="easy_sla_marginal")
        obs, _, _, info = self.env.step(
            Action(
                action_type="snooze",
                reasoning="Waiting for customer reply.",
                snooze_hours=2,
            )
        )
        self.assertGreater(obs.issue_age_hours, 72)
        self.assertEqual(info["reward_breakdown"]["snooze_sla_penalty"], -0.1)

    def test_flag_fraud_scores_correctly(self) -> None:
        self.env.reset(scenario_id="hard_fraud_chargeback")
        _, reward, _, _ = self.env.step(
            Action(
                action_type="flag_fraud",
                reasoning="Multiple rapid refund requests from different cards.",
                fraud_reason="Chargeback pattern consistent with card testing fraud.",
            )
        )
        self.assertGreater(reward, 0.0)

    def test_rule_baseline_runs_one_episode(self) -> None:
        scenario_id = "hard_old_invoice_question"
        observation = self.env.reset(scenario_id=scenario_id)
        agent = RuleBasedAgent()
        done = False
        info = {}
        while not done:
            observation, _, done, info = self.env.step(agent.next_action(observation))
        self.assertIn("final_score", info)


if __name__ == "__main__":
    unittest.main()
