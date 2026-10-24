"""Control-policy integrity and failure-budget regression checks."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'experiments/constrained_control'
sys.path.insert(0,str(PACKAGE/'scripts'))
from simulator import Reactor, initial_from_condition
from surrogate_model import load_models
import run_study


class ConstrainedControlTests(unittest.TestCase):
    def test_recorded_policy_has_same_numerical_solution(self):
        p=PACKAGE/'reference/results/policies/condition_0_target_1_seed_3101_mlp.json'
        r=json.loads(p.read_text())
        q=Reactor().simulate(r['controls'],r['initial'],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(q['terminal'],r['tight_verification']['terminal'],rtol=0,atol=1e-8)

    def test_portable_predictors_agree_with_saved_test_predictions(self):
        data=np.load(PACKAGE/'reference/data/test.npz',allow_pickle=False)
        for name,model in load_models(PACKAGE/'reference/results/locked_models.json').items():
            with self.subTest(model=name):
                expected=np.load(PACKAGE/f'reference/data/{name}_test_predictions.npz',allow_pickle=False)
                np.testing.assert_allclose(model.predict(data['X']),expected['prediction'],rtol=0,atol=1e-12)

    def test_failed_distinct_attempt_uses_budget_and_cached_failure_does_not_repeat(self):
        evaluator=run_study.Evaluator(initial_from_condition([1.,20.]),1.,1)
        with patch.object(run_study.reactor,'simulate',side_effect=RuntimeError('deliberate test failure')) as simulate:
            for _ in range(2):
                with self.assertRaises(run_study.SimulationFailure):
                    evaluator(np.zeros(7))
            self.assertEqual(simulate.call_count,1)
            self.assertEqual(len(evaluator.cache),1)
            self.assertEqual(len(evaluator.failed),1)
            with self.assertRaises(run_study.BudgetEnd):
                evaluator(np.full(7,.01))
            self.assertEqual(simulate.call_count,1)

    def test_study_cannot_overwrite_reference_or_existing_directory(self):
        for output in [PACKAGE/'reference',ROOT]:
            with self.subTest(output=str(output)):
                process=subprocess.run([sys.executable,str(PACKAGE/'run.py'),'study','--output',str(output)],capture_output=True,text=True)
                self.assertEqual(process.returncode,2)
                self.assertIn('Choose a new study directory',process.stderr)

    def test_cli_rejects_feed_exceeding_volume_constraint(self):
        process=subprocess.run([sys.executable,str(PACKAGE/'run.py'),'simulate','--feeds',*['0.2']*7],capture_output=True,text=True)
        self.assertEqual(process.returncode,2)
        self.assertIn('sum must be at most 0.65',process.stderr)


if __name__=='__main__':unittest.main()
