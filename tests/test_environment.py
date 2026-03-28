import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor

import anyio
import httpx
from fastapi.testclient import TestClient

from business_policy_env.baseline import RuleBasedAgent
from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action
from business_policy_env.rewards import shaped_reward
from business_policy_env.server import app
from business_policy_env.tasks import (
    build_ground_truth_payload,
    grade_actions,
    hard_components,
    medium_components,
    scenario_registry,
)


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
        self.assertEqual(reward, -0.1)
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
        self.assertLessEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["policy_violations"])
        self.assertEqual(info["reward_breakdown"]["policy_penalty"], -0.2)

    def test_dynamic_policy_shift_applies_mid_episode(self) -> None:
        observation = self.env.reset(scenario_id="medium_premier_same_day")
        self.assertEqual(observation.policy_version, "v1")
        self.assertTrue(observation.policy_shift_pending)

        self.env.step(
            Action(
                action_type="request_info",
                reasoning="Need details before resolution.",
                clarifying_question="Can you confirm your setup goals for today?",
            )
        )
        observation, _, _, info = self.env.step(
            Action(
                action_type="categorize",
                reasoning="This is onboarding help.",
                category="customer_success",
            )
        )

        self.assertEqual(observation.policy_version, "v2")
        self.assertFalse(observation.policy_shift_pending)
        self.assertIn("Policy update applied", info["policy_event"])

    def test_ambiguous_ticket_is_penalized_when_request_info_is_skipped(self) -> None:
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
        assert final_info is not None
        self.assertLess(final_info["component_scores"]["ambiguity_recognition"], 0.1)
        self.assertLess(final_info["final_score"], 0.5)

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
        self.assertIsNone(state_response.json()["ground_truth"])

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

    def test_concurrent_session_isolation(self) -> None:
        client = TestClient(app)

        def run_session(session_id: str, scenario_id: str, category: str) -> dict:
            client.post("/reset", json={"scenario_id": scenario_id}, headers={"X-Session-Id": session_id})
            client.post(
                "/step",
                json={
                    "action": {
                        "action_type": "categorize",
                        "reasoning": "Concurrent routing validation.",
                        "category": category,
                    }
                },
                headers={"X-Session-Id": session_id},
            )
            return client.get("/state", headers={"X-Session-Id": session_id}).json()

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(run_session, "session_1", "easy_vip_refund", "billing"),
                pool.submit(run_session, "session_2", "easy_vip_technical_issue", "technical_support"),
                pool.submit(run_session, "session_3", "easy_spam_detection", "spam"),
            ]
            states = [future.result() for future in futures]

        titles = {state["current_task_configuration"]["title"] for state in states}
        self.assertEqual(len(titles), 3)
        for state in states:
            self.assertEqual(state["internal_variables"]["steps_taken"], 1)

    def test_async_concurrent_session_isolation(self) -> None:
        async def run_concurrent_sessions() -> list[dict]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                async def run_session(session_id: str, scenario_id: str, category: str) -> dict:
                    await client.post("/reset", json={"scenario_id": scenario_id}, headers={"X-Session-Id": session_id})
                    await client.post(
                        "/step",
                        json={
                            "action": {
                                "action_type": "categorize",
                                "reasoning": "Async concurrent routing validation.",
                                "category": category,
                            }
                        },
                        headers={"X-Session-Id": session_id},
                    )
                    response = await client.get("/state", headers={"X-Session-Id": session_id})
                    return response.json()

                return await asyncio.gather(
                    run_session("async_session_1", "easy_vip_refund", "billing"),
                    run_session("async_session_2", "easy_vip_technical_issue", "technical_support"),
                    run_session("async_session_3", "easy_spam_detection", "spam"),
                )

        states = anyio.run(run_concurrent_sessions)
        titles = {state["current_task_configuration"]["title"] for state in states}
        self.assertEqual(len(titles), 3)
        for state in states:
            self.assertEqual(state["internal_variables"]["steps_taken"], 1)

    def test_policy_shift_and_fraud_interaction_enforced_after_shift(self) -> None:
        observation = self.env.reset(scenario_id="medium_policy_gaming_subtle")
        self.assertEqual(observation.policy_version, "v1")

        self.env.step(Action(action_type="categorize", reasoning="Billing route.", category="billing"))
        observation, _, _, info = self.env.step(
            Action(action_type="set_priority", reasoning="VIP handling.", priority="high")
        )
        self.assertEqual(observation.policy_version, "v2")
        self.assertIn("Policy update applied", info["policy_event"])

        _, _, _, info = self.env.step(
            Action(
                action_type="draft_response",
                reasoning="Attempt response without fraud flag after shift.",
                response_text="We are reviewing this request and will follow up shortly.",
            )
        )
        self.assertTrue(info["policy_violations"])
        self.assertEqual(info["reward_breakdown"]["policy_penalty"], -0.2)
        self.assertTrue(
            any(
                "Fraud indicators require flag_fraud before resolution actions." in item
                for item in info["policy_violations"]
            )
        )

    def test_policy_score_capped_after_any_violation(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        self.env.step(Action(action_type="set_priority", reasoning="Incorrect first pass.", priority="low"))
        self.env.step(Action(action_type="set_priority", reasoning="Correcting priority.", priority="high"))
        self.env.step(Action(action_type="categorize", reasoning="Billing route.", category="billing"))
        _, _, done, info = self.env.step(
            Action(
                action_type="escalate",
                reasoning="Refund exceeds threshold.",
                escalation_reason="Refund exceeds $500.",
            )
        )
        self.assertTrue(done)
        self.assertLessEqual(info["evaluation_metrics"]["policy_score"], 0.7)

    def test_state_can_optionally_include_ground_truth(self) -> None:
        client = TestClient(app)
        client.post("/reset", json={"scenario_id": "easy_vip_refund"})
        hidden_state = client.get("/state").json()
        full_state = client.get("/state?include_ground_truth=true").json()
        self.assertIsNone(hidden_state["ground_truth"])
        self.assertIsNotNone(full_state["ground_truth"])

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

    def test_cost_budget_penalizes_overspend(self) -> None:
        self.env.reset(scenario_id="hard_old_invoice_question")
        done = False
        info = {}
        while not done:
            _, _, done, info = self.env.step(
                Action(
                    action_type="draft_response",
                    reasoning="Intentionally expensive repetitive action for stress test.",
                    response_text="We are reviewing this now and will follow up shortly.",
                )
            )

        self.assertIn("cost_adjustment", info["reward_breakdown"])
        self.assertLess(info["reward_breakdown"]["cost_adjustment"], 0.0)

    def test_hidden_flags_are_not_visible_but_still_influence_truth(self) -> None:
        obs = self.env.reset(scenario_id="hard_hidden_fraud_delayed_detection")
        self.assertNotIn("fraud_risk", obs.account_flags)
        self.assertNotIn("chargeback_risk", obs.account_flags)
        state = self.env.state(include_ground_truth=True)
        self.assertTrue(state["ground_truth"]["expected_flag_fraud"])

    def test_consult_specialist_returns_feedback(self) -> None:
        self.env.reset(scenario_id="medium_policy_gaming_subtle")
        obs, _, _, _ = self.env.step(
            Action(
                action_type="consult_specialist",
                reasoning="Need specialist risk guidance.",
                specialist_question="Should we treat this as policy abuse or normal billing?",
            )
        )
        self.assertIsNotNone(obs.specialist_feedback)
        assert obs.specialist_feedback is not None
        self.assertIn("Specialist", obs.specialist_feedback)
        self.assertEqual(obs.current_email.direction, "system")

    def test_long_horizon_requires_min_steps_before_completion(self) -> None:
        self.env.reset(scenario_id="hard_long_horizon_escalation_chain")
        action_plan = [
            Action(
                action_type="request_info",
                reasoning="Need specifics before acting.",
                clarifying_question="Can you confirm invoice details and desired outcome?",
            ),
            Action(action_type="categorize", reasoning="Billing issue.", category="billing"),
            Action(action_type="set_priority", reasoning="High urgency due age and tier.", priority="urgent"),
            Action(
                action_type="consult_specialist",
                reasoning="Need specialist alignment.",
                specialist_question="Confirm escalation and timeline expectations.",
            ),
            Action(
                action_type="escalate",
                reasoning="Policy and severity require escalation.",
                escalation_reason="Long-running high-value billing failure.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Share clear next steps.",
                response_text=(
                    "We escalated this, assigned specialist ownership, and will provide "
                    "daily timeline updates."
                ),
            ),
        ]
        done = False
        for action in action_plan:
            _, _, done, _ = self.env.step(action)
        self.assertFalse(done)

        while not done:
            _, _, done, _ = self.env.step(
                Action(
                    action_type="draft_response",
                    reasoning="Continue status updates until closure threshold.",
                    response_text="Status update: specialist review remains active and timeline is being tracked.",
                )
            )
        self.assertGreaterEqual(self.env.state()["internal_variables"]["steps_taken"], 15)

    def test_adaptive_difficulty_selects_hard_after_strong_performance(self) -> None:
        self.env._performance_history = [0.91, 0.84, 0.88]
        observation = self.env.reset()
        self.assertEqual(observation.difficulty, "hard")

    def test_variation_seed_makes_reset_reproducible(self) -> None:
        scenario_id = "hard_hidden_risk_policy_shift"
        env_a = BusinessPolicyComplianceEnv(variation_seed=12345)
        env_b = BusinessPolicyComplianceEnv(variation_seed=12345)
        obs_a = env_a.reset(scenario_id=scenario_id)
        obs_b = env_b.reset(scenario_id=scenario_id)

        self.assertEqual(obs_a.issue_age_hours, obs_b.issue_age_hours)
        self.assertEqual(obs_a.refund_amount, obs_b.refund_amount)
        self.assertEqual(obs_a.account_flags, obs_b.account_flags)

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

    def test_keyword_stuffing_is_penalized_vs_balanced_response(self) -> None:
        scenario = scenario_registry()["medium_same_problem"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)

        common_prefix = [
            Action(
                action_type="request_info",
                reasoning="Need context.",
                clarifying_question="Can you share the exact app error and what changed?",
            ),
            Action(action_type="categorize", reasoning="Technical issue.", category="technical_support"),
            Action(action_type="set_priority", reasoning="Standard priority.", priority="medium"),
        ]

        stuffed = common_prefix + [
            Action(
                action_type="draft_response",
                reasoning="Keyword stuffed response.",
                response_text=("troubleshoot " * 40).strip(),
            )
        ]
        balanced = common_prefix + [
            Action(
                action_type="draft_response",
                reasoning="Balanced response.",
                response_text="We will troubleshoot the app update issue and send concrete next steps.",
            )
        ]

        stuffed_score = medium_components(stuffed, ground_truth)["response_appropriateness"]
        balanced_score = medium_components(balanced, ground_truth)["response_appropriateness"]
        self.assertLess(stuffed_score, balanced_score)

    def test_adversarial_failure_mode_is_reported(self) -> None:
        self.env.reset(scenario_id="medium_adversarial_refund_already_processed")
        self.env.step(Action(action_type="categorize", reasoning="Billing.", category="billing"))
        self.env.step(Action(action_type="set_priority", reasoning="Medium priority.", priority="medium"))
        _, _, done, info = self.env.step(
            Action(
                action_type="draft_response",
                reasoning="Incorrectly agreeing to duplicate refund.",
                response_text="We will process another refund immediately.",
            )
        )
        self.assertTrue(done)
        self.assertLess(info["evaluation_metrics"]["adversarial_resilience"], 0.55)
        self.assertIn("adversarial_miss", info["failure_modes"])

    def test_semantic_synonyms_receive_credit(self) -> None:
        scenario = scenario_registry()["medium_same_problem"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)
        actions = [
            Action(
                action_type="request_info",
                reasoning="Need context.",
                clarifying_question="Could you share the exact app error and when it began?",
            ),
            Action(action_type="categorize", reasoning="Technical issue.", category="technical_support"),
            Action(action_type="set_priority", reasoning="Standard priority.", priority="medium"),
            Action(
                action_type="draft_response",
                reasoning="Use synonym-rich but relevant response.",
                response_text="We will diagnose the app behavior after the latest patch and follow up promptly.",
            ),
        ]

        response_score = medium_components(actions, ground_truth)["response_appropriateness"]
        self.assertGreater(response_score, 0.5)

    def test_hard_response_prefers_thread_grounded_reply(self) -> None:
        scenario = scenario_registry()["hard_old_invoice_question"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)

        common_actions = [
            Action(action_type="categorize", reasoning="Billing.", category="billing"),
            Action(action_type="set_priority", reasoning="Aged ticket.", priority="urgent"),
        ]
        generic_actions = common_actions + [
            Action(
                action_type="draft_response",
                reasoning="Generic keyword response.",
                response_text="We will review the invoice and send an update.",
            )
        ]
        grounded_actions = common_actions + [
            Action(
                action_type="draft_response",
                reasoning="Thread-grounded response.",
                response_text=(
                    "We reviewed your earlier week follow-up on the March invoice fee and will provide "
                    "a dated update on the specific platform line item."
                ),
            )
        ]

        generic_score = hard_components(generic_actions, ground_truth)["response_completeness"]
        grounded_score = hard_components(grounded_actions, ground_truth)["response_completeness"]
        self.assertGreater(grounded_score, generic_score)

    def test_hard_response_action_consistency_detects_false_claims(self) -> None:
        scenario = scenario_registry()["hard_previous_agent_failed_escalation"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)

        inconsistent_actions = [
            Action(action_type="categorize", reasoning="Billing.", category="billing"),
            Action(action_type="set_priority", reasoning="Urgent.", priority="urgent"),
            Action(
                action_type="draft_response",
                reasoning="Incorrectly claims escalation happened.",
                response_text="We already escalated this with specialist confirmation.",
            ),
        ]
        consistent_actions = [
            Action(action_type="categorize", reasoning="Billing.", category="billing"),
            Action(action_type="set_priority", reasoning="Urgent.", priority="urgent"),
            Action(
                action_type="consult_specialist",
                reasoning="Need specialist confirmation.",
                specialist_question="Please confirm escalation routing and risk posture.",
            ),
            Action(
                action_type="escalate",
                reasoning="Escalate per policy.",
                escalation_reason="Refund exceeds $500.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Accurate update.",
                response_text="We already escalated this with specialist confirmation.",
            ),
        ]

        inconsistent_quality = hard_components(inconsistent_actions, ground_truth)["customer_quality"]
        consistent_quality = hard_components(consistent_actions, ground_truth)["customer_quality"]
        self.assertGreater(consistent_quality, inconsistent_quality)

    def test_hard_delayed_fraud_detection_is_penalized_in_grader(self) -> None:
        scenario = scenario_registry()["hard_hidden_fraud_delayed_detection"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)
        threshold = int(ground_truth["delayed_fraud_step_threshold"] or 1)

        timely_actions = [
            Action(
                action_type="flag_fraud",
                reasoning="Flag fraud immediately once signals are detected.",
                fraud_reason="Bypass and card-pattern signals indicate likely fraud.",
            )
        ]
        late_actions = [
            Action(
                action_type="draft_response",
                reasoning=f"Delay fraud decision step {idx + 1}.",
                response_text="We are reviewing this request.",
            )
            for idx in range(threshold)
        ]
        late_actions.append(
            Action(
                action_type="flag_fraud",
                reasoning="Fraud flagged late.",
                fraud_reason="Late fraud confirmation after multiple steps.",
            )
        )

        timely_fraud = hard_components(timely_actions, ground_truth)["fraud_handling"]
        late_fraud = hard_components(late_actions, ground_truth)["fraud_handling"]
        expected_late = round(threshold / (threshold + 1), 4)

        self.assertEqual(timely_fraud, 1.0)
        self.assertLess(late_fraud, timely_fraud)
        self.assertAlmostEqual(late_fraud, expected_late, places=4)

    def test_delayed_fraud_reward_penalty_is_proportional(self) -> None:
        scenario = scenario_registry()["hard_hidden_fraud_delayed_detection"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)

        actions_one_step_late = [
            Action(
                action_type="draft_response",
                reasoning="Delay step 1.",
                response_text="We are checking this.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Delay step 2.",
                response_text="We are still checking this.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Delay step 3.",
                response_text="Additional review in progress.",
            ),
            Action(
                action_type="flag_fraud",
                reasoning="Late fraud flag.",
                fraud_reason="Suspicious pattern identified.",
            ),
        ]
        breakdown_one_step_late = shaped_reward(
            actions_one_step_late,
            ground_truth,
            done=True,
            max_steps=scenario.max_steps,
            policy_violations=[],
            action_cost=0.0,
            cost_budget=scenario.cost_budget,
            snooze_crossed_sla=False,
            fraud_expected=True,
            policy_violation_seen=False,
        )
        self.assertAlmostEqual(breakdown_one_step_late.components["delayed_fraud_penalty"], -0.04, places=4)

        actions_many_steps_late = actions_one_step_late + [
            Action(
                action_type="draft_response",
                reasoning="Delay step 5.",
                response_text="Still under review.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Delay step 6.",
                response_text="Continuing review.",
            ),
        ]
        # Replace first fraud action to make first flag at step 6.
        actions_many_steps_late = [a for a in actions_many_steps_late if a.action_type != "flag_fraud"] + [
            Action(
                action_type="flag_fraud",
                reasoning="Very late fraud flag.",
                fraud_reason="Suspicious pattern identified late.",
            )
        ]
        breakdown_many_steps_late = shaped_reward(
            actions_many_steps_late,
            ground_truth,
            done=True,
            max_steps=scenario.max_steps,
            policy_violations=[],
            action_cost=0.0,
            cost_budget=scenario.cost_budget,
            snooze_crossed_sla=False,
            fraud_expected=True,
            policy_violation_seen=False,
        )
        self.assertAlmostEqual(breakdown_many_steps_late.components["delayed_fraud_penalty"], -0.12, places=4)

    def test_hard_response_action_consistency_detects_category_claim_mismatch(self) -> None:
        scenario = scenario_registry()["hard_old_invoice_question"]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)

        response_text = "We reviewed your refund charge and invoice payment issue and will update billing today."
        matched_actions = [
            Action(action_type="categorize", reasoning="Correct billing route.", category="billing"),
            Action(action_type="set_priority", reasoning="Urgent due age.", priority="urgent"),
            Action(
                action_type="draft_response",
                reasoning="Accurate billing-focused response.",
                response_text=response_text,
            ),
        ]
        mismatched_actions = [
            Action(
                action_type="categorize",
                reasoning="Incorrectly route as technical.",
                category="technical_support",
            ),
            Action(action_type="set_priority", reasoning="Urgent due age.", priority="urgent"),
            Action(
                action_type="draft_response",
                reasoning="Billing claim mismatches chosen category.",
                response_text=response_text,
            ),
        ]

        matched_quality = hard_components(matched_actions, ground_truth)["customer_quality"]
        mismatched_quality = hard_components(mismatched_actions, ground_truth)["customer_quality"]
        self.assertGreater(matched_quality, mismatched_quality)


if __name__ == "__main__":
    unittest.main()
