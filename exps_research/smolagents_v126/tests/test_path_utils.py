from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from exps_research.unified_framework.path_utils import bounded_artifact_path


class BoundedArtifactPathTest(unittest.TestCase):
    def test_preserves_short_name(self):
        with TemporaryDirectory() as temp_dir:
            path = bounded_artifact_path(temp_dir, "sample", "_scored.jsonl")
            self.assertEqual(Path(path).name, "sample_scored.jsonl")

    def test_shortens_deterministically(self):
        with TemporaryDirectory() as temp_dir:
            directory_length = len(str(Path(temp_dir).resolve()))
            limit = directory_length + 80
            base_name = "very_long_experiment_name_" * 10

            first = bounded_artifact_path(
                temp_dir,
                base_name,
                "_scored.jsonl",
                max_absolute_chars=limit,
            )
            second = bounded_artifact_path(
                temp_dir,
                base_name,
                "_scored.jsonl",
                max_absolute_chars=limit,
            )

            self.assertEqual(first, second)
            self.assertLessEqual(len(str(Path(first).resolve())), limit)
            self.assertTrue(Path(first).name.endswith("_scored.jsonl"))
            self.assertNotIn(base_name, Path(first).name)


if __name__ == "__main__":
    unittest.main()
