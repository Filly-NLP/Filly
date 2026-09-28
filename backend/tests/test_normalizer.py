import json
import shutil

import pytest

from app.services import normalizer as normalizer_service
from app.services.normalizer import FilipinoNormalizer, damerau_levenshtein_distance
from scripts.train_normalizer import DEFAULT_BASE_VOCABULARY, DEFAULT_PAIRS, build_artifact_bytes


def test_learned_normalization_and_known_words():
    normalizer = FilipinoNormalizer()

    assert normalizer.normalize_word("aq") == "ako"
    assert normalizer.normalize_word("tlga") == "talaga"
    assert normalizer.normalize_word("ako") == "ako"
    assert normalizer.normalize_word("kumain") == "kumain"
    assert normalizer.normalize_word("u") == "u"


def test_maximum_edit_distance_gates_only_candidates_over_the_limit():
    assert damerau_levenshtein_distance("aq", "ako") == 2
    assert FilipinoNormalizer(max_edit_distance=2).normalize_word("aq") == "ako"
    assert FilipinoNormalizer(max_edit_distance=1).normalize_word("aq") == "aq"
    with pytest.raises(ValueError, match="max_edit_distance"):
        FilipinoNormalizer(max_edit_distance=-1)


def test_sentence_edits_preserve_punctuation_case_and_original_offsets():
    normalizer = FilipinoNormalizer()

    normalized, changes = normalizer.normalize_text("AQ, po! nmn.")

    assert normalized == "AKO, po! naman."
    assert [(item.word, item.suggestion, item.start, item.end) for item in changes] == [
        ("AQ", "AKO", 0, 2),
        ("nmn", "naman", 8, 11),
    ]
    assert normalizer.normalize("AQ, po! nmn.") == changes


def test_valid_and_unsupported_mixed_case_words_keep_their_original_surface():
    normalizer = FilipinoNormalizer()

    for word in ("iPhone", "McDo", "aKo"):
        assert normalizer.normalize_word(word) == word

    assert normalizer.normalize_text("iPhone McDo aKo") == ("iPhone McDo aKo", [])


def test_urls_email_numbers_unknown_words_and_unicode_are_safe():
    normalizer = FilipinoNormalizer()
    source = "🙂 aq https://example.com/aq aq@example.com aq2 2026 qzxv"

    normalized, changes = normalizer.normalize_text(source)

    assert normalized == "🙂 ako https://example.com/aq aq@example.com aq2 2026 qzxv"
    assert [(item.word, item.start, item.end) for item in changes] == [("aq", 2, 4)]
    assert normalizer.normalize_text("") == ("", [])
    assert normalizer.normalize_text("   2026!") == ("   2026!", [])


def test_damerau_levenshtein_supports_transpositions_and_nonlocal_edits():
    assert damerau_levenshtein_distance("form", "from") == 1
    assert damerau_levenshtein_distance("ca", "abc") == 2


def test_candidate_ranking_computes_each_distance_once_and_keeps_stable_ties(monkeypatch):
    normalizer = FilipinoNormalizer()
    normalizer.vocabulary = {"first", "second"}
    monkeypatch.setattr(
        normalizer,
        "_collect_candidates",
        lambda *_args: {"first": 0.8, "second": 0.7},
    )
    compared = []

    def tied_distance(_source, candidate):
        compared.append(candidate)
        return 1

    monkeypatch.setattr(normalizer_service, "damerau_levenshtein_distance", tied_distance)

    assert normalizer._normalize_lowercase("source") == ("first", 1)
    assert compared == ["first", "second"]


def test_artifact_build_is_deterministic_and_training_only():
    first = build_artifact_bytes(DEFAULT_PAIRS, DEFAULT_BASE_VOCABULARY)
    second = build_artifact_bytes(DEFAULT_PAIRS, DEFAULT_BASE_VOCABULARY)

    assert first == second
    metadata = json.loads(first["metadata.json"])
    assert metadata["training_data"]["pair_count"] == 303
    assert metadata["algorithm_source"]["base_vocabulary_member"].endswith("with_info.txt")
    assert "test_words.csv" not in metadata["training_data"]["path"]


def test_configured_artifact_paths_load_colocated_metadata_and_fail_closed(tmp_path, monkeypatch):
    default_artifacts = FilipinoNormalizer()._resource_signature[0].parent
    staged = tmp_path / "configured-normalizer"
    staged.mkdir()
    rules_path = staged / "custom-rules.json"
    vocabulary_path = staged / "custom-vocabulary.txt"
    for source, destination in (
        (default_artifacts / "rules.json", rules_path),
        (default_artifacts / "vocabulary.txt", vocabulary_path),
        (default_artifacts / "metadata.json", staged / "metadata.json"),
    ):
        shutil.copyfile(source, destination)

    normalizer = FilipinoNormalizer(rules_path=rules_path, vocabulary_path=vocabulary_path)
    assert normalizer.normalize_word("aq") == "ako"
    assert normalizer._resource_signature[2] == (staged / "metadata.json").resolve()
    single_override = FilipinoNormalizer(rules_path=rules_path)
    assert single_override._resource_signature[1] == (default_artifacts / "vocabulary.txt").resolve()
    assert single_override._resource_signature[2] == (staged / "metadata.json").resolve()

    monkeypatch.setattr(normalizer_service, "_normalizer", None)
    cached = normalizer_service.get_normalizer(
        rules_path=rules_path,
        vocabulary_path=vocabulary_path,
    )
    assert normalizer_service.get_normalizer() is cached

    other = tmp_path / "other-normalizer"
    other.mkdir()
    for filename in ("rules.json", "vocabulary.txt", "metadata.json"):
        shutil.copyfile(default_artifacts / filename, other / filename)
    with pytest.raises(RuntimeError, match="different resource paths"):
        normalizer_service.get_normalizer(
            rules_path=other / "rules.json",
            vocabulary_path=other / "vocabulary.txt",
        )

    vocabulary_path.write_bytes(vocabulary_path.read_bytes() + b"corrupt\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FilipinoNormalizer(rules_path=rules_path, vocabulary_path=vocabulary_path)
