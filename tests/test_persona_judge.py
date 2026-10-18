import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import experiments as exp


def result(predicted='B'):
    return json.dumps(dict(predicted=predicted, confidence=4,
                           tone_match=dict(A=2, B=5, C=1), reason='Warm and conversational.'))


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def record(persona='friendly_advisor', words=10, **overrides):
    row = dict(model='test-model', suite='models', condition='persona=' + persona,
               case_id='budget_bali_upgrade', repeat=0, persona_key=persona,
               persona_version='v1', query='Traveler question', final_answer='Aria ' + 'word ' * words,
               stopped_reason='final_answer', error=None, simulated=False, tools_used=[],
               use_cot=True, few_shot=True, instruction_placement='system')
    row.update(overrides)
    return row


class TestPersonaJudge(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / 'judge.jsonl'
        self.path_patch = patch.object(exp, 'PERSONA_JUDGE_LOG', self.log)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.client = Mock()
        self.client.chat.completions.create.return_value = response(result())
        self.client_patch = patch.object(exp.core, 'OpenAI', return_value=self.client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def test_name_blinding_whole_words(self):
        self.assertEqual(exp.blind_persona_names('ARIA, Marcus and eLeNa; Marian ariatic'),
                         '[name], [name] and [name]; Marian ariatic')

    def test_valid_json(self):
        self.assertEqual(exp.validate_persona_judgment(result())['confidence'], 4)

    def test_invalid_json_and_ranges(self):
        good = json.loads(result())
        variants = ['not JSON', '[]']
        for field, value in [('predicted', 'D'), ('confidence', True), ('confidence', 6),
                             ('tone_match', dict(A=0, B=2, C=3)), ('tone_match', dict(A=2)),
                             ('reason', '')]:
            bad = {**good, field: value}
            variants.append(json.dumps(bad))
        for content in variants:
            with self.subTest(content=content), self.assertRaises((ValueError, TypeError)):
                exp.validate_persona_judgment(content)

    def test_retry_once_and_mapping(self):
        self.client.chat.completions.create.side_effect = [response('{}'), response(result())]
        rows = exp.judge_persona_fidelity([record()])
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.assertEqual(rows[0]['predicted_persona'], 'friendly_advisor')
        self.assertEqual(rows[0]['tone_match'], 5)
        self.assertIsNone(rows[0]['error'])

    def test_retry_exhaustion_error_row(self):
        self.client.chat.completions.create.return_value = response('{}')
        rows = exp.judge_persona_fidelity([record()])
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.assertEqual(rows[0]['error'], 'invalid_json_after_retry')
        self.assertIsNone(rows[0]['predicted_persona'])

    def test_resume_skips_success_and_errors(self):
        self.client.chat.completions.create.return_value = response('{}')
        exp.judge_persona_fidelity([record()])
        before = self.log.read_bytes()
        self.client.reset_mock()
        self.assertEqual(exp.judge_persona_fidelity([record()]), [])
        self.client.chat.completions.create.assert_not_called()
        self.assertEqual(before, self.log.read_bytes())

    def test_prompt_blind_and_profiles_verbatim(self):
        exp.judge_persona_fidelity([record(final_answer='I am ARIA; Marcus agrees with Elena.')])
        call = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(call['temperature'], 0)
        self.assertEqual(call['response_format'], {'type': 'json_object'})
        messages = call['messages']
        self.assertEqual(json.loads(messages[1]['content'])['answer'],
                         'I am [name]; [name] agrees with [name].')
        for forbidden in ('test-model', 'friendly_advisor', 'budget_bali_upgrade', 'condition'):
            self.assertNotIn(forbidden, str(messages))
        for profile in exp.core.PERSONA_PROFILES.values():
            self.assertIn(profile['tone'], messages[0]['content'])
            self.assertIn(profile['task'], messages[0]['content'])

    def test_fixed_label_mapping_all_personas(self):
        self.assertEqual(list(exp.JUDGE_LABELS.values()),
                         ['creative_curator', 'friendly_advisor', 'strict_auditor'])
        for label, key in exp.JUDGE_LABELS.items():
            self.client.chat.completions.create.return_value = response(result(label))
            row = exp.judge_persona_fidelity([record(persona=key)])[0]
            self.assertEqual(row['predicted_persona'], key)

    def test_candidate_filter_and_latest_attempt(self):
        rows = [record(), record(error='error'), record(simulated=True, repeat=1),
                record(stopped_reason='forced_answer', repeat=2), record(suite='prompting', repeat=3)]
        self.assertEqual(exp.judge_persona_fidelity(rows), [])
        self.client.chat.completions.create.assert_not_called()

    def test_length_baseline_and_offline_summary(self):
        records = [record('strict_auditor', 2), record('friendly_advisor', 5), record('creative_curator', 9)]
        original = copy.deepcopy(records)
        rows = exp.judge_persona_fidelity(records)
        self.assertEqual(records, original)
        summary = exp.summarize_persona_judge()
        self.assertEqual(summary['accuracy'][0]['length_accuracy_pct'], 100)
        self.assertEqual(summary['accuracy'][0]['accuracy_pct'], 100/3)
        self.assertEqual(summary['judged'], 3)
        self.assertEqual(len(self.log.read_text(encoding='utf-8').splitlines()), 3)

    def test_incomplete_triplet_no_baseline(self):
        exp.judge_persona_fidelity([record()])
        self.assertEqual(exp.summarize_persona_judge()['accuracy'][0]['baseline_n'], 0)

    def test_api_error_does_not_log_secret(self):
        self.client.chat.completions.create.side_effect = RuntimeError('secret-do-not-log')
        row = exp.judge_persona_fidelity([record()])[0]
        self.assertEqual(row['error'], 'api_error:RuntimeError')
        self.assertNotIn('secret-do-not-log', self.log.read_text(encoding='utf-8'))

    def test_load_both_logs(self):
        with patch.object(exp, 'load_records', return_value=[]) as load:
            self.assertEqual(exp.load_persona_judge_inputs(), [])
        self.assertEqual(load.call_count, 2)
        self.assertEqual({c.kwargs['path'].name for c in load.call_args_list},
                         {'c4_experiments.jsonl', 'c4_experiments_bedrock.jsonl'})

    def test_v2_summary_and_self_judge_exclusion(self):
        exp.judge_persona_fidelity([
            record(model='gpt-4o-mini'),
            record(suite='persona_v2', model='haiku', persona_version='v1'),
            record(suite='persona_v2', model='haiku', persona_version='v2', condition='v2')])
        summary = exp.summarize_persona_judge()
        self.assertEqual(summary['overall']['n'], 3)
        self.assertEqual(summary['excluding_gpt4o_mini']['n'], 0)
        self.assertEqual(summary['overall_excluding_gpt4o_mini']['n'], 2)
        self.assertEqual(summary['persona_v2'][0]['n'], 1)
        self.assertEqual(summary['persona_v2'][1]['n'], 1)


if __name__ == '__main__':
    unittest.main()
