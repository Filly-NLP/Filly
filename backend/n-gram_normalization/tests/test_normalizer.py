import tempfile
import unittest
from pathlib import Path

from ngram_normalizer import load_model, normalize_text, save_model, train_model


class NormalizerWorkflowTest(unittest.TestCase):
    def test_train_save_load_and_normalize(self):
        pairs = [("tlga", "talaga"), ("kc", "kasi"), ("ako", "ako")]
        model = train_model(pairs, vocabulary_path=None)
        with tempfile.TemporaryDirectory() as directory:
            path = save_model(model, Path(directory) / "model.json")
            loaded = load_model(path)
        self.assertEqual(normalize_text("Tlga kc ako!", loaded), "Talaga kasi ako!")


if __name__ == "__main__":
    unittest.main()
