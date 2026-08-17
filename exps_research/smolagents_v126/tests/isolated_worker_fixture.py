from exps_research.unified_framework.processors.base import ExperimentProcessor


class IsolatedWorkerFixtureProcessor(ExperimentProcessor):
    """Lightweight spawn-safe processor used by isolation tests."""

    def create_model(self, worker_id=0, use_local_model=False, use_single_endpoint=False):
        return {"worker_id": worker_id}

    def process_entry(self, entry, model, **kwargs):
        behavior = entry.get("behavior", "return")
        if behavior == "hang":
            while True:
                pass
        if behavior == "error":
            raise RuntimeError("fixture failure")
        return {
            "model_id": self.model_kwargs["model_id"],
            "question": entry["question"],
            "generated_answer": entry.get("answer", "ok"),
            "true_answer": entry.get("answer"),
            "log_data": {
                "trajectory_steps": [],
                "metadata": {"state": "success", "success": True},
            },
            "input_tokens": 1,
            "output_tokens": 1,
        }
