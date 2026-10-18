"""Unit tests for Assignment C4 ReAct Agent and LangGraph Workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import react_core as core


class ReActAgentTests(unittest.TestCase):
    def test_tool_search_travel_packages(self):
        res = core.search_travel_packages("bali")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["packages"][0]["name"], "Bali Bliss")

    def test_tool_calculate_trip_budget(self):
        # Bali Bliss is $1,850 pp. For 2 travelers = $3,700.
        res = core.calculate_trip_budget("Bali Bliss", 2)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["estimated_total_usd"], "$3,700")

        # With upgrades: deluxe room ($250*2=$500) + private guide ($400) = $4,600
        upgraded = core.calculate_trip_budget("Bali Bliss", 2, room_upgrade=True, private_guide=True)
        self.assertEqual(upgraded["estimated_total_usd"], "$4,600")

    def test_tool_lookup_policy_rule(self):
        flight_policy = core.lookup_policy_rule("flights")
        self.assertEqual(flight_policy["status"], "found")
        self.assertIn("excluded", flight_policy["policy"].lower())

        unknown_policy = core.lookup_policy_rule("pet policy")
        self.assertEqual(unknown_policy["status"], "unknown_policy")

    def test_parse_react_output_action(self):
        raw = (
            "Thought: I need to verify package availability.\n"
            "Action: search_travel_packages(query=\"japan\")\n"
        )
        thought, action, params, answer = core.parse_react_output(raw)
        self.assertEqual(thought, "I need to verify package availability.")
        self.assertEqual(action, "search_travel_packages")
        self.assertEqual(params, {"query": "japan"})
        self.assertIsNone(answer)

    def test_parse_react_output_final_answer(self):
        raw = (
            "Thought: I have the calculation ready.\n"
            "Final Answer: The estimated cost for Bali Bliss is $3,700 for two travelers."
        )
        thought, action, params, answer = core.parse_react_output(raw)
        self.assertEqual(thought, "I have the calculation ready.")
        self.assertIsNone(action)
        self.assertIsNone(params)
        self.assertIn("$3,700", answer)

    def test_persona_profiles_defined(self):
        self.assertIn("friendly_advisor", core.PERSONA_PROFILES)
        self.assertIn("strict_auditor", core.PERSONA_PROFILES)
        self.assertIn("creative_curator", core.PERSONA_PROFILES)

        # Check parameter divergence
        friendly_temp = core.PERSONA_PROFILES["friendly_advisor"]["llm_config"]["temperature"]
        strict_temp = core.PERSONA_PROFILES["strict_auditor"]["llm_config"]["temperature"]
        self.assertGreater(friendly_temp, strict_temp)

    def test_langgraph_workflow_compilation(self):
        graph = core.build_react_graph()
        self.assertIsNotNone(graph)

    def test_langgraph_state_machine_structure(self):
        """Verify the compiled LangGraph nodes and edges match the architectural specification.

        Guarantees docs/react_graph.svg and C4_README.md cannot silently drift from the implementation.
        """
        graph = core.react_graph.get_graph()
        expected_nodes = {"__start__", "reason", "act", "guard", "force_answer", "__end__"}
        self.assertEqual(set(graph.nodes.keys()), expected_nodes)

        expected_edges = {
            # Solid unconditional edges (conditional=False)
            ("__start__", "reason", False),
            ("act", "reason", False),
            ("guard", "__end__", False),
            # Conditional edges routed by should_continue / should_guard_forced (conditional=True)
            ("reason", "act", True),
            ("reason", "reason", True),
            ("reason", "guard", True),
            ("reason", "force_answer", True),
            ("reason", "__end__", True),
            ("force_answer", "guard", True),
            ("force_answer", "__end__", True),
        }
        actual_edges = {(e.source, e.target, e.conditional) for e in graph.edges}
        self.assertEqual(actual_edges, expected_edges)

    def test_should_continue_logic(self):
        # If final answer is present, must terminate
        state_done = {"final_answer": "Finished", "action_name": None, "iteration": 1}
        self.assertEqual(core.should_continue(state_done), core.END)

        # If action is requested, must route to act
        state_act = {"final_answer": None, "action_name": "search_travel_packages", "iteration": 1}
        self.assertEqual(core.should_continue(state_act), "act")

        # If iteration limit reached, must terminate
        state_limit = {"final_answer": None, "action_name": "search_travel_packages", "iteration": 5}
        self.assertEqual(core.should_continue(state_limit), core.END)


class ParserRobustnessTests(unittest.TestCase):
    def test_parentheses_inside_string_argument(self):
        raw = 'Thought: log it\nAction: record_customer_lead(name="Dana", email="d@x.com", notes="Rome (honeymoon), 2 pax")'
        _, action, params, _ = core.parse_react_output(raw)
        self.assertEqual(action, "record_customer_lead")
        self.assertEqual(params["notes"], "Rome (honeymoon), 2 pax")

    def test_hallucinated_observation_is_discarded(self):
        raw = (
            "Thought: need price\nAction: calculate_trip_budget(package_name=\"Bali Bliss\", num_travelers=2)\n"
            "Observation: {\"estimated_total_usd\": \"$9,999\"}\nFinal Answer: It costs $9,999."
        )
        _, action, _, answer = core.parse_react_output(raw)
        self.assertEqual(action, "calculate_trip_budget")
        self.assertIsNone(answer)

    def test_markdown_and_positional_and_json_bools(self):
        raw = '**Thought:** check\n**Action:** calculate_trip_budget("Bali Bliss", 2, room_upgrade=true)'
        thought, action, params, _ = core.parse_react_output(raw)
        self.assertEqual(thought, "check")
        self.assertEqual(action, "calculate_trip_budget")
        self.assertEqual(params, {"room_upgrade": True, "__positional__": ["Bali Bliss", 2]})

    def test_thought_only_is_format_error(self):
        self.assertEqual(core.parse_react_output("Thought: still thinking"), ("still thinking", None, None, None))
        state = {"final_answer": None, "action_name": None, "format_error": True, "iteration": 1}
        self.assertEqual(core.should_continue(state), "reason")

    def test_plain_text_is_final_answer(self):
        self.assertEqual(core.parse_react_output("Hello!")[3], "Hello!")


class ProviderAndPromptTests(unittest.TestCase):
    def test_explicit_provider_wins_over_other_keys(self):
        env = {"LLM_PROVIDER": "openrouter", "GEMINI_API_KEY": "g-key", "OPENROUTER_API_KEY": "o-key", "REACT_SIMULATE": ""}
        with patch.dict(core.os.environ, env):
            self.assertEqual(core.get_active_provider(), "openrouter")

    def test_placeholder_keys_mean_simulated(self):
        env = {"LLM_PROVIDER": "", "GEMINI_API_KEY": "your_gemini_api_key_here",
               "OPENROUTER_API_KEY": "", "OPENAI_API_KEY": "", "REACT_SIMULATE": "",
               "AWS_BEARER_TOKEN_BEDROCK": "", "AWS_ACCESS_KEY_ID": "", "AWS_PROFILE": ""}
        with patch.dict(core.os.environ, env):
            self.assertEqual(core.get_active_provider(), "simulated")

    def test_live_failure_is_reported_not_simulated(self):
        with patch.object(core, "get_active_provider", return_value="openai"), \
             patch.object(core, "_call_llm", side_effect=core.LLMError("429 quota")):
            res = core.run_react_agent("Bali for 2?")
        self.assertEqual(res["stopped_reason"], "llm_error")
        self.assertIn("429", res["error"])
        self.assertEqual(res["steps"], [])

    def test_prompt_toggles(self):
        state = {"user_query": "hi", "persona_key": "strict_auditor", "react_steps": [],
                 "few_shot": False, "use_cot": False, "instruction_placement": "user",
                 "conversation_history": [{"role": "user", "content": "Bali?"}]}
        system, user = core.build_prompt_parts(state)
        self.assertIsNone(system)
        self.assertNotIn("WORKED EXAMPLE", user)
        self.assertIn("Do not write Thought lines", user)
        self.assertIn("Traveler: Bali?", user)
        state.update(few_shot=True, instruction_placement="system")
        system, user = core.build_prompt_parts(state)
        self.assertIn("WORKED EXAMPLE", system)
        self.assertNotIn("PERSONA", user)

    def test_llm_overrides(self):
        cfg = core.resolve_llm_config("strict_auditor", {"temperature": 0.9, "top_k": 3})
        self.assertEqual((cfg["temperature"], cfg["top_k"], cfg["top_p"]), (0.9, 3, 0.5))
        with self.assertRaises(ValueError):
            core.resolve_llm_config("strict_auditor", {"temprature": 0.9})


class SimulatedLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_patch = patch.object(core, "LOG_DIR", Path(self.tmp.name))
        self.log_patch.start()
        self.env_patch = patch.dict(core.os.environ, {"REACT_SIMULATE": "1"})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.log_patch.stop()
        self.tmp.cleanup()

    def test_budget_loop_uses_real_observation(self):
        res = core.run_react_agent("Calculate budget for 2 people on Swiss Alpine with private guide")
        self.assertTrue(res["simulated"])
        self.assertEqual(res["tools_used"], ["calculate_trip_budget"])
        self.assertIn("$4,700", res["final_answer"])

    def test_unknown_policy_is_logged(self):
        res = core.run_react_agent("Can I get a refund if I cancel?")
        self.assertEqual(res["tools_used"], ["lookup_policy_rule", "record_unanswered_inquiry"])
        self.assertTrue((Path(self.tmp.name) / "c4_feedback.jsonl").exists())


class QuotaHandlingTests(unittest.TestCase):
    def test_daily_quota_fails_fast_without_retry(self):
        err = Exception("429 RESOURCE_EXHAUSTED. {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
                        "'quotaValue': '20'}, {'retryDelay': '11220s'}")
        client = MagicMock()
        client.models.generate_content.side_effect = err
        with patch.object(core.genai, "Client", return_value=client), \
             patch.dict(core.os.environ, {"GEMINI_API_KEY": "k"}), \
             patch.object(core.time, "sleep") as sleep:
            with self.assertRaises(core.LLMError) as ctx:
                core._call_gemini(None, "hi", core.resolve_llm_config("strict_auditor"), "gemini-x-flash")
        self.assertIn("daily quota", str(ctx.exception))
        self.assertIn("20 requests/day", str(ctx.exception))
        sleep.assert_not_called()
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_harness_stops_after_consecutive_errors(self):
        import experiments as exp
        failing = {"error": "quota", "stopped_reason": "llm_error", "final_answer": "x", "tools_used": [],
                   "iterations": 1, "latency_s": 0.0}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(exp, "EXPERIMENT_LOG", Path(tmp) / "exp.jsonl"), \
             patch.object(exp.core, "run_react_agent", return_value=failing):
            records = exp.run_suite("personas", verbose=False)
        self.assertEqual(len(records), 3)


class BedrockTests(unittest.TestCase):
    MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def setUp(self):
        core._BEDROCK_REJECTED.clear()
        self.client = MagicMock()
        self.client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Thought: ok\nFinal Answer: hi"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        self.patches = [
            patch.object(core, "_bedrock_client", return_value=self.client),
            patch.dict(core.os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "t", "AWS_REGION": "us-east-1"}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        core._BEDROCK_REJECTED.clear()

    def test_request_shape_and_ignored_penalties(self):
        cfg = core.resolve_llm_config("strict_auditor")
        text, meta = core._call_bedrock("SYSTEM", "USER", cfg, self.MODEL)
        req = self.client.converse.call_args.kwargs
        self.assertEqual(text, "Thought: ok\nFinal Answer: hi")
        self.assertEqual(req["system"], [{"text": "SYSTEM"}])
        self.assertEqual(req["inferenceConfig"]["temperature"], 0.1)
        self.assertEqual(req["inferenceConfig"]["topP"], 0.5)
        self.assertEqual(req["inferenceConfig"]["stopSequences"], core.STOP_SEQUENCES)
        self.assertEqual(req["additionalModelRequestFields"], {"top_k": 10})
        self.assertEqual(meta["ignored_params"], ["frequency_penalty", "presence_penalty"])

    def test_rejected_top_p_is_dropped_remembered_and_recorded(self):
        self.client.converse.side_effect = [
            Exception("An error occurred (ValidationException): `temperature` and `top_p` cannot both be specified"),
            self.client.converse.return_value,
            self.client.converse.return_value,
        ]
        cfg = core.resolve_llm_config("friendly_advisor")
        _, meta = core._call_bedrock(None, "USER", cfg, self.MODEL)
        self.assertNotIn("topP", self.client.converse.call_args.kwargs["inferenceConfig"])
        self.assertIn("top_p", meta["ignored_params"])
        self.assertNotIn("top_p", meta["applied_config"])
        core._call_bedrock(None, "USER", cfg, self.MODEL)  # remembered: no second rejection
        self.assertEqual(self.client.converse.call_count, 3)

    def test_temperature_above_bedrock_max_is_clamped_and_recorded(self):
        cfg = core.resolve_llm_config("strict_auditor", {"temperature": 1.2})
        _, meta = core._call_bedrock(None, "USER", cfg, self.MODEL)
        self.assertEqual(self.client.converse.call_args.kwargs["inferenceConfig"]["temperature"], 1.0)
        self.assertEqual(meta["applied_config"]["temperature"], 1.0)
        self.assertIn("1.2 -> 1.0", meta["adjusted_params"]["temperature"])

    def test_access_denied_is_explained(self):
        self.client.converse.side_effect = Exception("An error occurred (AccessDeniedException): no access")
        with self.assertRaises(core.LLMError) as ctx:
            core._call_bedrock(None, "USER", core.resolve_llm_config("strict_auditor"), self.MODEL)
        self.assertIn("Model access", str(ctx.exception))

    def test_provider_detection(self):
        env = {"LLM_PROVIDER": "", "GEMINI_API_KEY": "", "OPENROUTER_API_KEY": "", "OPENAI_API_KEY": "",
               "REACT_SIMULATE": "", "AWS_BEARER_TOKEN_BEDROCK": "t"}
        with patch.dict(core.os.environ, env):
            self.assertEqual(core.get_active_provider(), "bedrock")
        with patch.dict(core.os.environ, {**env, "LLM_PROVIDER": "bedrock", "BEDROCK_MODEL_ID": ""}):
            self.assertEqual(core.get_active_model(), core.DEFAULT_MODELS["bedrock"])

    def test_summary_keeps_models_apart(self):
        import experiments as exp
        base = {"suite": "personas", "condition": "c", "run_id": "r", "simulated": False, "error": None,
                "iterations": 2, "format_errors": 0, "latency_s": 1.0,
                "metrics": {"passed": True, "tools_ok": True, "facts_ok": True, "hallucination_hits": [], "answer_words": 5}}
        rows = [{**base, "provider": "openrouter", "model": "m1"}, {**base, "provider": "bedrock", "model": "m2"}]
        table = exp.summarize(rows)
        self.assertIn("| m1 | personas | c | 1 |", table)
        self.assertIn("| m2 | personas | c | 1 |", table)


class GuardrailTests(unittest.TestCase):
    CANCEL = "What happens if I need to cancel my Santorini trip because of a storm?"

    def _run(self, replies, guardrails, query=None):
        meta = {"applied_config": {}, "ignored_params": []}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(core, "LOG_DIR", Path(tmp)), \
             patch.object(core, "get_active_provider", return_value="openai"), \
             patch.object(core, "_call_llm", side_effect=[(r, meta) for r in replies]):
            result = core.run_react_agent(query or self.CANCEL, "friendly_advisor", guardrails=guardrails)
            logged = (Path(tmp) / "c4_feedback.jsonl").exists()
        return result, logged

    SKIPPED_LOGGING = [
        'Thought: check policy\nAction: lookup_policy_rule(topic="cancellation")',
        "Thought: nothing approved\nFinal Answer: We have no published terms. I've passed your question to our team.",
    ]

    def test_required_logging_is_done_by_the_guard(self):
        result, logged = self._run(self.SKIPPED_LOGGING, guardrails=True)
        self.assertTrue(logged)
        self.assertIn("record_unanswered_inquiry", result["tools_used"])
        self.assertTrue(any("record_unanswered_inquiry" in e for e in result["guardrail_events"]))

    def test_guard_is_off_by_default_so_experiments_measure_the_model(self):
        result, logged = self._run(self.SKIPPED_LOGGING, guardrails=False)
        self.assertFalse(logged)
        self.assertEqual(result["tools_used"], ["lookup_policy_rule"])
        self.assertEqual(result["guardrail_events"], [])

    def test_guard_does_not_double_log(self):
        replies = [
            'Thought: check\nAction: lookup_policy_rule(topic="cancellation")',
            f'Thought: log it\nAction: record_unanswered_inquiry(question="{self.CANCEL}")',
            "Thought: done\nFinal Answer: Logged for our team.",
        ]
        result, logged = self._run(replies, guardrails=True)
        self.assertTrue(logged)
        self.assertEqual(result["tools_used"].count("record_unanswered_inquiry"), 1)
        self.assertEqual(result["guardrail_events"], [])

    def test_airfare_quote_is_removed_but_package_prices_stay(self):
        replies = [
            'Thought: check\nAction: lookup_policy_rule(topic="flights")',
            "Thought: done\nFinal Answer: Flights are excluded. Round-trip airfare to Rome is about $600–$1,200 per person. "
            "Our Rome & Amalfi Escape is $1,650 per person.",
        ]
        result, _ = self._run(replies, guardrails=True, query="Do you book flights to Rome, and how much is airfare?")
        self.assertNotIn("$600", result["final_answer"])
        self.assertIn("$1,650", result["final_answer"])
        self.assertIn("Flights are excluded.", result["final_answer"])
        self.assertIn(core.AIRFARE_DISCLAIMER, result["final_answer"])
        self.assertTrue(any("airfare" in e for e in result["guardrail_events"]))

    def test_clean_answer_is_left_unchanged(self):
        replies = ['Thought: price\nAction: calculate_trip_budget(package_name="Bali Bliss", num_travelers=2)',
                   "Thought: done\nFinal Answer: Bali Bliss for two is $3,700. Flights are not included."]
        result, logged = self._run(replies, guardrails=True, query="Bali for 2?")
        self.assertEqual(result["final_answer"], "Bali Bliss for two is $3,700. Flights are not included.")
        self.assertFalse(logged)
        self.assertEqual(result["guardrail_events"], [])


class OpenRouterRetryTests(unittest.TestCase):
    def _client(self, responses):
        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        return client

    def test_http200_error_body_with_429_is_retried(self):
        empty = MagicMock(choices=None, error={"message": "temporarily rate-limited upstream", "code": 429})
        ok = MagicMock()
        ok.choices = [MagicMock(message=MagicMock(content="Final Answer: hi"))]
        client = self._client([empty, ok])
        with patch.object(core, "OpenAI", return_value=client), patch.object(core.time, "sleep"), \
             patch.object(core, "_openrouter_supported_params", return_value=None), \
             patch.dict(core.os.environ, {"OPENROUTER_API_KEY": "k"}):
            text, _ = core._call_openai_compatible("openrouter", None, "hi",
                                                   core.resolve_llm_config("strict_auditor"), "m:free")
        self.assertEqual(text, "Final Answer: hi")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_http200_error_body_without_retryable_code_fails(self):
        empty = MagicMock(choices=None, error={"message": "invalid model", "code": 400})
        client = self._client([empty])
        with patch.object(core, "OpenAI", return_value=client), patch.object(core.time, "sleep"), \
             patch.object(core, "_openrouter_supported_params", return_value=None), \
             patch.dict(core.os.environ, {"OPENROUTER_API_KEY": "k"}):
            with self.assertRaises(core.LLMError):
                core._call_openai_compatible("openrouter", None, "hi", core.resolve_llm_config("strict_auditor"), "m:free")


class ResumeTests(unittest.TestCase):
    def test_resume_skips_done_runs_and_rescore_keeps_latest(self):
        import experiments as exp
        ok = {"error": None, "stopped_reason": "final_answer", "final_answer": "4,200 kyoto flight", "tools_used": [],
              "iterations": 1, "latency_s": 0.0, "model": "m", "provider": "p", "simulated": False, "format_errors": 0}
        calls = []

        def fake_run(query, **kwargs):
            calls.append(query)
            return ok

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(exp, "EXPERIMENT_LOG", Path(tmp) / "exp.jsonl"), \
             patch.object(exp.core, "get_active_model", return_value="m"), \
             patch.object(exp.core, "run_react_agent", side_effect=fake_run):
            # One earlier failed attempt and one success for the same condition
            with exp.EXPERIMENT_LOG.open("w", encoding="utf-8") as f:
                base = {"model": "m", "suite": "personas", "condition": "persona=friendly_advisor", "repeat": 0,
                        "run_id": "r0", "provider": "p", "simulated": False, "iterations": 1, "format_errors": 0,
                        "latency_s": 0.0, "tools_used": [], "stopped_reason": "final_answer", "final_answer": "x"}
                f.write(json.dumps({**base, "case_id": "budget_bali_upgrade", "error": "boom"}) + "\n")
                f.write(json.dumps({**base, "case_id": "budget_swiss_guide", "error": None}) + "\n")
            exp.run_suite("personas", case_ids=["budget_bali_upgrade", "budget_swiss_guide"], verbose=False, resume=True)
            # 3 personas x 2 cases = 6, minus the one successful logged run
            self.assertEqual(len(calls), 5)
            records = exp.rescore(exp.load_records())
            keys = [(r["condition"], r["case_id"]) for r in records]
            self.assertEqual(len(keys), len(set(keys)))
            bali = [r for r in records if r["case_id"] == "budget_bali_upgrade" and r["condition"] == "persona=friendly_advisor"]
            self.assertIsNone(bali[0]["error"])  # the retry replaced the failed attempt


class ModelsSuiteTests(unittest.TestCase):
    def test_models_suite_runs_every_persona_on_every_model(self):
        import experiments as exp
        seen = []

        def fake_run(query, **kwargs):
            seen.append((kwargs["llm_overrides"]["model"], kwargs["persona_key"]))
            return {"error": None, "stopped_reason": "final_answer", "final_answer": "ok", "tools_used": [],
                    "iterations": 1, "latency_s": 0.0, "model": kwargs["llm_overrides"]["model"], "provider": "p",
                    "simulated": False, "format_errors": 0}

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(exp, "EXPERIMENT_LOG", Path(tmp) / "exp.jsonl"), \
             patch.dict(exp.core.os.environ, {"EXPERIMENT_MODELS": "m1, m2"}), \
             patch.object(exp.core, "run_react_agent", side_effect=fake_run):
            exp.run_suite("models", case_ids=["lead_capture"], verbose=False)
        self.assertEqual(len(seen), 2 * len(core.PERSONA_PROFILES))
        self.assertEqual({m for m, _ in seen}, {"m1", "m2"})

    def test_models_suite_requires_models_and_is_not_in_all(self):
        import experiments as exp
        with patch.dict(exp.core.os.environ, {"EXPERIMENT_MODELS": ""}):
            with self.assertRaises(ValueError):
                exp.models_conditions("friendly_advisor")
        self.assertNotIn("models", exp.CORE_SUITES)


class ScorerTests(unittest.TestCase):
    def test_negated_claim_is_not_a_hallucination(self):
        import experiments as exp
        pattern = r"group tours? (are|is) (planned|scheduled|available)"
        self.assertFalse(exp._asserted_matches(pattern, "no group tour is planned for this itinerary."))
        self.assertTrue(exp._asserted_matches(pattern, "good news: group tours are planned for 2027."))
        airfare = exp.AIRFARE_QUOTE
        self.assertTrue(exp._asserted_matches(airfare, "flights are excluded. a typical flight to rome is about $600"))
        # A negated mention early in the sentence must not hide the asserted quote later on
        self.assertTrue(exp._asserted_matches(airfare, "i don't have live flight pricing, but typical round-trip "
                                                       "airfare to rome ranges widely, anywhere from $400-$1,200"))
        self.assertFalse(exp._asserted_matches(airfare, "flights are excluded from the rome package ($1,650 per person)"))

    def test_wrong_grand_total_is_caught_but_breakdowns_are_not(self):
        import experiments as exp
        wrong = exp._wrong_total("4,700", subtotal="4,300")
        self.assertTrue(exp._asserted_matches(wrong, "estimated total: $4,700 - guide: +$400 **grand total: $5,100**"))
        self.assertFalse(exp._asserted_matches(wrong, "**total for 2 travelers:** $4,300 - guide +$400 = $4,700"))
        self.assertFalse(exp._asserted_matches(wrong, "comes to $4,700 total ($2,150 base per person + $400 guide)"))


class ToolFixTests(unittest.TestCase):
    def test_policy_aliases(self):
        self.assertEqual(core.lookup_policy_rule("refund")["topic"], "cancellation")
        self.assertEqual(core.lookup_policy_rule("airfare")["topic"], "flights")

    def test_multi_word_search_and_fuzzy_package(self):
        self.assertEqual(core.search_travel_packages("bali honeymoon")["packages"][0]["name"], "Bali Bliss")
        self.assertEqual(core.calculate_trip_budget("Swiss Alps", 1)["package"], "Swiss Alpine Discovery")
        self.assertEqual(core.calculate_trip_budget("Bali", 2, room_upgrade="false")["estimated_total_usd"], "$3,700")


class PersonaRationaleTests(unittest.TestCase):
    def test_every_persona_has_rationale_for_all_llm_config_keys(self):
        for pkey, p in core.PERSONA_PROFILES.items():
            self.assertIn("config_rationale", p, f"Persona '{pkey}' missing 'config_rationale' dict")
            rationale = p["config_rationale"]
            for key in p["llm_config"]:
                self.assertIn(key, rationale, f"Persona '{pkey}' missing rationale for key '{key}'")
                self.assertIsInstance(rationale[key], str)
                self.assertTrue(len(rationale[key].strip()) > 10, f"Persona '{pkey}' rationale for '{key}' is too short")

    def test_marcus_and_elena_key_rationales_match_requirements(self):
        marcus_rat = core.PERSONA_PROFILES["strict_auditor"]["config_rationale"]
        elena_rat = core.PERSONA_PROFILES["creative_curator"]["config_rationale"]
        self.assertIn("0.1", marcus_rat["temperature"])
        self.assertTrue(any(w in marcus_rat["temperature"].lower() for w in ("deterministic", "exact", "repeatable", "factual")))
        self.assertIn("0.5", elena_rat["presence_penalty"])
        self.assertTrue(any(w in elena_rat["presence_penalty"].lower() for w in ("new", "detail", "sights", "experiences", "repeating")))


class InstructionsAblationTests(unittest.TestCase):
    def test_instruction_constants_composition(self):
        expected = f"{core.INSTRUCTION_FORMAT}\n\n{core.INSTRUCTION_OPERATIONAL_RULES}"
        self.assertEqual(core.INSTRUCTION_RULES, expected)

    def test_prompt_byte_for_byte_identity_in_full_mode(self):
        for pkey in core.PERSONA_PROFILES:
            state_default: core.ReActAgentState = {
                "user_query": "How much for Bali?",
                "persona_key": pkey,
                "use_cot": True,
                "few_shot": True,
                "instruction_placement": "system",
            }
            state_explicit_full: core.ReActAgentState = {
                "user_query": "How much for Bali?",
                "persona_key": pkey,
                "use_cot": True,
                "few_shot": True,
                "instruction_placement": "system",
                "instructions_mode": "full",
            }
            sys_def, user_def = core.build_prompt_parts(state_default)
            sys_full, user_full = core.build_prompt_parts(state_explicit_full)
            self.assertEqual(sys_def, sys_full)
            self.assertEqual(user_def, user_full)
            # Ensure full operational rules and worked examples are in the prompt
            self.assertIn("OPERATIONAL RULES:", sys_full)
            self.assertIn("WORKED EXAMPLE:", sys_full)
            self.assertIn("SECOND EXAMPLE:", sys_full)
            self.assertIn(core.TOOL_DESCRIPTIONS, sys_full)

    def test_format_only_mode_drops_rules_and_examples_keeps_format_and_tools(self):
        state_format: core.ReActAgentState = {
            "user_query": "How much for Bali?",
            "persona_key": "strict_auditor",
            "use_cot": True,
            "few_shot": True,
            "instruction_placement": "system",
            "instructions_mode": "format_only",
        }
        sys_prompt, user_msg = core.build_prompt_parts(state_format)
        self.assertIn("INPUT YOU RECEIVE:", sys_prompt)
        self.assertIn("OUTPUT YOU MUST RETURN", sys_prompt)
        self.assertIn(core.TOOL_DESCRIPTIONS, sys_prompt)
        self.assertIn("REASONING DIRECTIVE:", sys_prompt)
        self.assertNotIn("OPERATIONAL RULES:", sys_prompt)
        self.assertNotIn("WORKED EXAMPLE:", sys_prompt)
        self.assertNotIn("SECOND EXAMPLE:", sys_prompt)

    def test_run_react_agent_validates_instructions_mode(self):
        with self.assertRaises(ValueError):
            core.run_react_agent("test query", instructions_mode="invalid_mode")

    def test_run_react_agent_stores_instructions_mode(self):
        with patch("react_core.get_active_provider", return_value="simulated"):
            res_full = core.run_react_agent("test query", instructions_mode="full")
            self.assertEqual(res_full["instructions_mode"], "full")
            res_format = core.run_react_agent("test query", instructions_mode="format_only")
            self.assertEqual(res_format["instructions_mode"], "format_only")

    def test_instructions_suite_in_experiments(self):
        import experiments as exp
        self.assertIn("instructions", exp.SUITES)
        self.assertNotIn("instructions", exp.CORE_SUITES)
        conds = exp.instructions_conditions("friendly_advisor")
        self.assertEqual(len(conds), 6)  # 3 personas x 2 modes
        labels = [label for label, _ in conds]
        self.assertIn("persona=strict_auditor | instructions=full", labels)
        self.assertIn("persona=strict_auditor | instructions=format_only", labels)


class LoopGuardTests(unittest.TestCase):
    def test_loop_guards_off_by_default(self):
        with patch("react_core.get_active_provider", return_value="simulated"):
            res = core.run_react_agent("test query")
            self.assertFalse(res["loop_guards"])
            self.assertEqual(res["loop_guard_events"], [])

    def test_repeat_action_blocked_when_loop_guards_on(self):
        initial_steps = [
            {"type": "action", "content": "lookup_policy_rule(topic='flights')"},
            {"type": "observation", "content": json.dumps({"status": "found", "policy": "No flights"})},
        ]
        state: core.ReActAgentState = {
            "action_name": "lookup_policy_rule",
            "action_params": {"topic": "flights"},
            "react_steps": list(initial_steps),
            "loop_guards": True,
            "loop_guard_events": [],
        }
        res = core.act_node(state)
        self.assertEqual(len(res["react_steps"]), len(initial_steps) + 2)
        obs_step = res["react_steps"][-1]
        self.assertEqual(obs_step["type"], "observation")
        obs_data = json.loads(obs_step["content"])
        self.assertEqual(obs_data["status"], "error")
        self.assertIn("already called lookup_policy_rule", obs_data["message"])
        self.assertEqual(len(res["loop_guard_events"]), 1)
        self.assertIn("Blocked repeat call", res["loop_guard_events"][0])

    def test_repeat_action_allowed_when_loop_guards_off(self):
        initial_steps = [
            {"type": "action", "content": "lookup_policy_rule(topic='flights')"},
            {"type": "observation", "content": json.dumps({"status": "found", "policy": "No flights"})},
        ]
        state: core.ReActAgentState = {
            "action_name": "lookup_policy_rule",
            "action_params": {"topic": "flights"},
            "react_steps": list(initial_steps),
            "loop_guards": False,
            "loop_guard_events": [],
        }
        res = core.act_node(state)
        obs_step = res["react_steps"][-1]
        obs_data = json.loads(obs_step["content"])
        self.assertEqual(obs_data["status"], "found")
        self.assertEqual(res["loop_guard_events"], [])

    def test_scripted_llm_identical_behavior_when_loop_guards_off(self):
        scripted_outputs = [
            "Thought: Check budget.\nAction: calculate_trip_budget(package_name=\"Bali Bliss\", num_travelers=2)",
            "Thought: Got budget.\nFinal Answer: It is $3,700 total.",
        ]
        call_idx = 0
        def fake_call_llm(system, user, cfg, provider, model):
            nonlocal call_idx
            out = scripted_outputs[call_idx]
            call_idx += 1
            return out, {"applied_config": {}, "ignored_params": []}

        with patch("react_core._call_llm", side_effect=fake_call_llm):
            with patch("react_core.get_active_provider", return_value="openai"):
                res_off = core.run_react_agent("How much for Bali?", loop_guards=False)
                self.assertEqual(res_off["stopped_reason"], "final_answer")
                self.assertIn("$3,700", res_off["final_answer"])
                self.assertEqual(res_off["iterations"], 2)
                self.assertEqual(res_off["loop_guard_events"], [])

    def test_repeat_action_blocked_with_different_argument_order_and_formatting(self):
        initial_steps = [
            {"type": "action", "content": 'calculate_trip_budget(num_travelers=2, package_name="Bali Bliss")'},
            {"type": "observation", "content": json.dumps({"status": "success", "estimated_total_usd": "$3,700"})},
        ]
        state: core.ReActAgentState = {
            "action_name": "calculate_trip_budget",
            "action_params": {"package_name": "Bali Bliss", "num_travelers": 2},
            "react_steps": list(initial_steps),
            "loop_guards": True,
            "loop_guard_events": [],
        }
        res = core.act_node(state)
        obs_step = res["react_steps"][-1]
        obs_data = json.loads(obs_step["content"])
        self.assertEqual(obs_data["status"], "error")
        self.assertIn("already called calculate_trip_budget", obs_data["message"])
        self.assertEqual(len(res["loop_guard_events"]), 1)

    def test_loop_guards_off_exact_node_sequence_at_step_limit(self):
        visited_nodes = []
        orig_reason = core.reason_node
        orig_act = core.act_node
        orig_force = core.force_answer_node

        def tracked_reason(s):
            visited_nodes.append("reason")
            return orig_reason(s)

        def tracked_act(s):
            visited_nodes.append("act")
            return orig_act(s)

        def tracked_force(s):
            visited_nodes.append("force_answer")
            return orig_force(s)

        scripted_loop = "Thought: Checking.\nAction: lookup_policy_rule(topic='flights')"
        with patch.object(core, "reason_node", side_effect=tracked_reason), \
             patch.object(core, "act_node", side_effect=tracked_act), \
             patch.object(core, "force_answer_node", side_effect=tracked_force), \
             patch("react_core._call_llm", return_value=(scripted_loop, {"applied_config": {}, "ignored_params": []})), \
             patch("react_core.get_active_provider", return_value="openai"):
            wf = core.build_react_graph()
            with patch.object(core, "react_graph", wf):
                res = core.run_react_agent("test query", loop_guards=False)
                # 5 iterations: reason -> act -> reason -> act -> reason -> act -> reason -> act -> reason -> END
                self.assertEqual(visited_nodes, ["reason", "act", "reason", "act", "reason", "act", "reason", "act", "reason"])
                self.assertNotIn("force_answer", visited_nodes)
                self.assertEqual(res["stopped_reason"], "max_iterations")

    def test_forced_final_answer_at_step_limit(self):
        loop_call = "Thought: Need policy.\nAction: lookup_policy_rule(topic='flights')"
        forced_reply = "Thought: Step limit reached.\nFinal Answer: Flights are completely excluded."
        calls = []

        def fake_call_llm(system, user, cfg, provider, model):
            calls.append(user)
            if len(calls) <= 5:
                return loop_call, {"applied_config": {}, "ignored_params": []}
            return forced_reply, {"applied_config": {}, "ignored_params": []}

        with patch("react_core._call_llm", side_effect=fake_call_llm):
            with patch("react_core.get_active_provider", return_value="openai"):
                res = core.run_react_agent("Do you book flights?", loop_guards=True)
                self.assertEqual(res["stopped_reason"], "forced_answer")
                self.assertIn("Flights are completely excluded.", res["final_answer"])
                self.assertIn(core.FORCE_ANSWER_PROMPT, calls[-1])
                self.assertTrue(len(res["loop_guard_events"]) > 0)

    def test_forced_final_answer_fallback_to_canned_if_model_fails_to_answer(self):
        loop_call = "Thought: Need policy.\nAction: lookup_policy_rule(topic='flights')"
        calls = []

        def fake_call_llm(system, user, cfg, provider, model):
            calls.append(user)
            return loop_call, {"applied_config": {}, "ignored_params": []}

        with patch("react_core._call_llm", side_effect=fake_call_llm):
            with patch("react_core.get_active_provider", return_value="openai"):
                res = core.run_react_agent("Do you book flights?", loop_guards=True)
                self.assertEqual(res["stopped_reason"], "max_iterations")
                self.assertIn("I couldn't finish reasoning within 5 steps", res["final_answer"])

    def test_loop_guards_suite_in_experiments(self):
        import experiments as exp
        self.assertIn("loop_guards", exp.SUITES)
        self.assertNotIn("loop_guards", exp.CORE_SUITES)
        conds = exp.loop_guards_conditions("friendly_advisor")
        self.assertEqual(len(conds), 2)  # loop_guards=off, loop_guards=on
        labels = [label for label, _ in conds]
        self.assertIn("loop_guards=off", labels)
        self.assertIn("loop_guards=on", labels)

    def test_score_run_treats_forced_answer_like_final_answer(self):
        import experiments as exp
        case = exp.TEST_CASES[3]  # policy_flights
        res_forced = {
            "stopped_reason": "forced_answer",
            "tools_used": ["lookup_policy_rule"],
            "final_answer": "We do not book flights to Rome; international airfare is excluded.",
        }
        metrics = exp.score_run(case, res_forced)
        self.assertTrue(metrics["completed"])
        self.assertTrue(metrics["passed"])


class PersonaConflictTests(unittest.TestCase):
    def test_check_persona_conflicts_structure_and_findings(self):
        conflicts = core.check_persona_conflicts()
        self.assertIsInstance(conflicts, list)
        self.assertGreater(len(conflicts), 0)
        personas_with_conflicts = {c[0] for c in conflicts}
        self.assertIn(core.PERSONA_PROFILES["friendly_advisor"]["name"], personas_with_conflicts)
        self.assertIn(core.PERSONA_PROFILES["creative_curator"]["name"], personas_with_conflicts)
        self.assertNotIn(core.PERSONA_PROFILES["strict_auditor"]["name"], personas_with_conflicts)

        for persona, phrase, rule, evidence in conflicts:
            self.assertTrue(len(persona) > 0)
            self.assertTrue(len(phrase) > 0)
            self.assertTrue(len(rule) > 0)
            self.assertTrue(len(evidence) > 0)

    def test_check_persona_conflicts_v2_has_no_conflicts(self):
        conflicts_v2 = core.check_persona_conflicts(version="v2")
        self.assertEqual(len(conflicts_v2), 0)


class PersonaV2Tests(unittest.TestCase):
    def test_v1_prompt_byte_for_byte_identical_to_default(self):
        for pkey in core.PERSONA_PROFILES:
            state_default: core.ReActAgentState = {
                "user_query": "Test query",
                "persona_key": pkey,
                "use_cot": True,
                "few_shot": True,
                "instruction_placement": "system",
            }
            state_v1: core.ReActAgentState = {
                **state_default,
                "persona_version": "v1",
            }
            sys_def, user_def = core.build_prompt_parts(state_default)
            sys_v1, user_v1 = core.build_prompt_parts(state_v1)
            self.assertEqual(sys_def, sys_v1)
            self.assertEqual(user_def, user_v1)

    def test_v2_differs_only_in_persona_line(self):
        for pkey in core.PERSONA_PROFILES:
            state_v1: core.ReActAgentState = {
                "user_query": "Test query",
                "persona_key": pkey,
                "use_cot": True,
                "few_shot": True,
                "instruction_placement": "system",
                "persona_version": "v1",
            }
            state_v2: core.ReActAgentState = {
                **state_v1,
                "persona_version": "v2",
            }
            sys_v1, user_v1 = core.build_prompt_parts(state_v1)
            sys_v2, user_v2 = core.build_prompt_parts(state_v2)
            self.assertEqual(user_v1, user_v2)

            if pkey == "strict_auditor":
                self.assertEqual(sys_v1, sys_v2)
            else:
                v1_tone = core.PERSONA_PROFILES[pkey]["prompt_tone_instructions"]
                v2_tone = core.PERSONA_PROFILES[pkey]["prompt_tone_instructions_v2"]
                self.assertNotEqual(v1_tone, v2_tone)
                self.assertEqual(sys_v1.replace(v1_tone, v2_tone), sys_v2)

    def test_persona_version_validation_and_result(self):
        with self.assertRaises(ValueError):
            core.run_react_agent("test", persona_version="invalid")

        with patch("react_core.get_active_provider", return_value="simulated"):
            res_v1 = core.run_react_agent("test", persona_version="v1")
            self.assertEqual(res_v1["persona_version"], "v1")
            res_v2 = core.run_react_agent("test", persona_version="v2")
            self.assertEqual(res_v2["persona_version"], "v2")


class PersonaRoutingTests(unittest.TestCase):
    def test_routing_keyword_rules(self):
        # Strict auditor keywords: budget, cost, price, refund, policy, visa, flight
        self.assertEqual(core.route_persona("What is the trip budget?"), "strict_auditor")
        self.assertEqual(core.route_persona("How much does it cost?"), "strict_auditor")
        self.assertEqual(core.route_persona("Is there a discount on the price?"), "strict_auditor")
        self.assertEqual(core.route_persona("Can I get a refund if cancelled?"), "strict_auditor")
        self.assertEqual(core.route_persona("What is your cancellation policy?"), "strict_auditor")
        self.assertEqual(core.route_persona("Do you help obtain a tourist visa?"), "strict_auditor")
        self.assertEqual(core.route_persona("Do you book international flights?"), "strict_auditor")

        # Creative curator keywords: highlights, experience, culture, food, what is it like
        self.assertEqual(core.route_persona("Tell me about the highlights of Japan"), "creative_curator")
        self.assertEqual(core.route_persona("What is the cultural experience like?"), "creative_curator")
        self.assertEqual(core.route_persona("Tell me about Italian food tasting"), "creative_curator")
        self.assertEqual(core.route_persona("What is it like visiting the Alps?"), "creative_curator")

        # Fallback -> friendly_advisor
        self.assertEqual(core.route_persona("Hello, I would like to plan a trip with Dana."), "friendly_advisor")
        self.assertEqual(core.route_persona("I want a honeymoon consultation."), "friendly_advisor")

    def test_auto_persona_in_run_react_agent(self):
        with patch("react_core.get_active_provider", return_value="simulated"):
            res = core.run_react_agent("Calculate the total budget for 2 people", persona_key="auto")
            self.assertEqual(res["persona_requested"], "auto")
            self.assertEqual(res["persona_key"], "strict_auditor")
            self.assertEqual(res["persona"], core.PERSONA_PROFILES["strict_auditor"]["name"])
            self.assertIn("budget", res["persona_route_reason"])

            res_fb = core.run_react_agent("I'd like a consultation. Name: Dana, dana@x.com", persona_key="auto")
            self.assertEqual(res_fb["persona_requested"], "auto")
            self.assertEqual(res_fb["persona_key"], "friendly_advisor")
            self.assertIn("fallback", res_fb["persona_route_reason"])


class FisherExactTests(unittest.TestCase):
    def test_fisher_exact_against_hand_computed_value(self):
        import experiments as exp
        # 12/12 vs 12/18: p = comb(18, 6) / comb(30, 6) = 18564 / 593775 = 0.03126437...
        p = exp.fisher_exact_one_sided(12, 12, 12, 18)
        self.assertAlmostEqual(p, 0.03126437, places=5)

    def test_fisher_exact_boundaries(self):
        import experiments as exp
        self.assertEqual(exp.fisher_exact_one_sided(0, 0, 0, 0), 1.0)
        self.assertEqual(exp.fisher_exact_one_sided(5, 5, 5, 5), 1.0)


class NewSuitesTests(unittest.TestCase):
    def test_persona_v2_suite_conditions_and_defaults(self):
        import experiments as exp
        self.assertIn("persona_v2", exp.SUITES)
        self.assertNotIn("persona_v2", exp.CORE_SUITES)
        conds = exp.persona_v2_conditions("friendly_advisor")
        self.assertEqual(len(conds), 4)  # 2 personas x 2 versions
        labels = [c[0] for c in conds]
        self.assertIn("persona=friendly_advisor | version=v1", labels)
        self.assertIn("persona=friendly_advisor | version=v2", labels)
        self.assertIn("persona=creative_curator | version=v1", labels)
        self.assertIn("persona=creative_curator | version=v2", labels)

    def test_auto_persona_suite_conditions(self):
        import experiments as exp
        self.assertIn("auto_persona", exp.SUITES)
        self.assertNotIn("auto_persona", exp.CORE_SUITES)
        conds = exp.auto_persona_conditions("friendly_advisor")
        self.assertEqual(len(conds), 2)
        labels = [c[0] for c in conds]
        self.assertIn("persona=auto", labels)
        self.assertIn("persona=friendly_advisor", labels)

    def test_model_override_applied_to_conditions(self):
        import experiments as exp
        conds = [("base", {"persona_key": "friendly_advisor"}), ("custom", {"persona_key": "strict_auditor", "llm_overrides": {"model": "m_custom"}})]
        applied = []
        def fake_run(query, **kwargs):
            applied.append(kwargs.get("llm_overrides", {}).get("model"))
            return {"error": None, "stopped_reason": "final_answer", "final_answer": "ok", "tools_used": [],
                    "iterations": 1, "latency_s": 0.0, "model": kwargs.get("llm_overrides", {}).get("model", "def"),
                    "provider": "p", "simulated": False, "format_errors": 0}

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(exp, "EXPERIMENT_LOG", Path(tmp) / "exp.jsonl"), \
             patch.object(exp.core, "run_react_agent", side_effect=fake_run):
            exp.run_conditions("test_suite", conds, [exp.TEST_CASES[0]], model_override="m_override", verbose=False)
        self.assertEqual(applied, ["m_override", "m_custom"])


if __name__ == "__main__":
    unittest.main()
