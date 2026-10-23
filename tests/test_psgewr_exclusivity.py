import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from autode.lpsge.psgewr import EvalFn, PSGEWR, _exclusive_training


class PSGEWRExclusivityTests(unittest.TestCase):
    def search(self):
        return PSGEWR(EvalFn(lambda individual, seed: [0.], [0.]), None, 4)

    def test_concurrent_instance_is_rejected_before_setup(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []
        first, second = self.search(), self.search()
        def training():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Test did not release training")
        def worker():
            try:
                first.run()
            except BaseException as exc:
                errors.append(exc)
        with patch.object(first, "_run", side_effect=training), \
             patch.object(second, "_run", return_value="completed") as run:
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    second.run()
                run.assert_not_called()
            finally:
                release.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(second.run(), "completed")

    def test_exception_releases_lock(self):
        search = self.search()
        with patch.object(search, "_run", side_effect=ValueError("training failed")):
            with self.assertRaisesRegex(ValueError, "training failed"):
                search.run()
        with patch.object(search, "_run", return_value="completed"):
            self.assertEqual(search.run(), "completed")

    def test_other_process_cannot_start_training(self):
        code = '''
from autode.lpsge.psgewr import _exclusive_training
try:
    with _exclusive_training():
        raise AssertionError("Concurrent training was allowed")
except RuntimeError as exc:
    assert "already active" in str(exc)
'''
        with _exclusive_training():
            result = subprocess.run([sys.executable, "-c", code],
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
