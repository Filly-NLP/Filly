"""Integration checks for the retained CLI adapters."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_ROOT.parent
CLI_ROOT = BACKEND_ROOT / "n-gram_normalization"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.normalizer import FilipinoNormalizer  # noqa: E402


class NormalizerCliIntegrationTest(unittest.TestCase):
    def run_cli(self, script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_normalize_cli_matches_current_backend(self) -> None:
        source = "aq pra kc ndi skul nya nalang nandun"
        expected, _changes = FilipinoNormalizer().normalize_text(source)
        artifact_dir = str(BACKEND_ROOT / "artifacts" / "normalizer")

        first = self.run_cli(CLI_ROOT / "normalize.py", source, "--model", artifact_dir, "--cutoff", "100")
        second = self.run_cli(CLI_ROOT / "normalize.py", source, "--artifact-dir", artifact_dir)

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout.strip(), expected)
        self.assertEqual(second.stdout.strip(), expected)

    def test_train_cli_builds_loadable_deterministic_artifacts_outside_live_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.csv"
            vocabulary = root / "vocabulary.txt"
            pairs.write_text("Input,Target\nprang,parang\n", encoding="utf-8")
            vocabulary.write_text("{'word': 'parang'}\n", encoding="latin1")
            outputs = [root / "first", root / "second"]

            for output in outputs:
                result = self.run_cli(
                    CLI_ROOT / "train_normalizer.py",
                    "--data",
                    str(pairs),
                    "--vocab",
                    str(vocabulary),
                    "--max-sub",
                    "2",
                    "--output",
                    str(output),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Built normalizer artifacts", result.stdout)

            for filename in ("rules.json", "vocabulary.txt", "metadata.json"):
                first = hashlib.sha256((outputs[0] / filename).read_bytes()).hexdigest()
                second = hashlib.sha256((outputs[1] / filename).read_bytes()).hexdigest()
                self.assertEqual(first, second, filename)

            loaded = FilipinoNormalizer(outputs[0])
            self.assertEqual(loaded.max_ngram, 2)
            self.assertEqual(loaded.candidate_cutoff, 100)


if __name__ == "__main__":
    unittest.main()
