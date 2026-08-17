import threading
import time
import unittest

from exps_research.unified_framework.processors.base import ExperimentProcessor


class RecordingProcessor(ExperimentProcessor):
    def __init__(self):
        super().__init__({"model_id": "fake"})
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.active_models = set()
        self.reused_model_concurrently = False

    def create_models(self, max_workers, use_local_model=False, use_single_endpoint=False):
        return [object() for _ in range(max_workers)]

    def process_entry(self, entry, model, **kwargs):
        model_key = id(model)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if model_key in self.active_models:
                self.reused_model_concurrently = True
            self.active_models.add(model_key)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
            self.active_models.remove(model_key)
        return {"question": entry["question"]}


class BoundedThreadPoolTest(unittest.TestCase):
    def test_limits_in_flight_work_and_reuses_only_free_model_slots(self):
        processor = RecordingProcessor()
        entries = [{"question": str(index)} for index in range(25)]

        results = processor.process_dataset(
            entries,
            max_workers=4,
            use_process_pool=False,
        )

        self.assertEqual(len(results), len(entries))
        self.assertLessEqual(processor.max_active, 4)
        self.assertFalse(processor.reused_model_concurrently)


if __name__ == "__main__":
    unittest.main()
