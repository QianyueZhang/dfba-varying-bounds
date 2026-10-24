"""Check command selection without executing or modifying scientific studies."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("case_runner", Path(__file__).resolve().parents[1] / "tools/run_case.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class CaseRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["first.py", "second.py"]:
            (self.root / name).write_text("pass\n")
        (self.root / "cases.json").write_text(json.dumps({"demo": {
            "description": "Test only", "commands": [["first.py"], ["second.py", "--value", "2"]],
            "output_directories": ["results"],
        }}))
        self.patch = patch.object(RUNNER, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            RUNNER.main(args)

    def test_preview_has_no_execution_or_output_side_effect(self):
        with patch.object(RUNNER.subprocess, "run") as run:
            self.invoke(["demo"])
            run.assert_not_called()
        self.assertFalse((self.root / "results").exists())

    def test_step_executes_only_selected_command(self):
        with patch.object(RUNNER.subprocess, "run") as run:
            self.invoke(["demo", "--step", "2", "--execute"])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][1:], ["second.py", "--value", "2"])
            self.assertEqual(run.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
        self.assertTrue((self.root / "results").is_dir())

    def test_invalid_steps_fail_before_execution(self):
        for step in ["0", "-1", "3"]:
            with self.subTest(step=step), patch.object(RUNNER.subprocess, "run") as run:
                with self.assertRaises(SystemExit) as error:
                    self.invoke(["demo", "--step", step, "--execute"])
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_missing_case_is_an_error_when_execution_requested(self):
        with self.assertRaises(SystemExit):
            self.invoke(["--execute"])

    def test_missing_script_fails_before_execution(self):
        (self.root / "second.py").unlink()
        with patch.object(RUNNER.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                self.invoke(["demo", "--execute"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
