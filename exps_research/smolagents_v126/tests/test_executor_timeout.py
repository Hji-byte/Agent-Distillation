import time
import unittest

from smolagents.local_python_executor import ExecutionTimeoutError, timeout


class ExecutorTimeoutTest(unittest.TestCase):
    def test_timeout_returns_without_waiting_for_worker_completion(self):
        @timeout(0.05)
        def slow_call():
            time.sleep(0.5)

        started = time.monotonic()
        with self.assertRaises(ExecutionTimeoutError):
            slow_call()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.3)

    def test_wrapped_exception_is_preserved(self):
        @timeout(1)
        def fail():
            raise RuntimeError("expected")

        with self.assertRaisesRegex(RuntimeError, "expected"):
            fail()


if __name__ == "__main__":
    unittest.main()
