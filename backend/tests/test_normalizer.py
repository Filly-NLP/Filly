import json
import shutil
import warnings
from pathlib import Path

import pytest

from app.services import normalizer as normalizer_service
from app.services.normalizer import FilipinoNormalizer, damerau_levenshtein_distance
from scripts.train_normalizer import (
    DEFAULT_BASE_VOCABULARY,
    DEFAULT_PAIRS,
    build_artifact_bytes,
    build_curated_mappings,
)


def test_learned_normalization_and_known_words():
    normalizer = FilipinoNormalizer()

    assert normalizer.normalize_word("aq") == "ako"
    assert normalizer.normalize_word("tlga") == "talaga"
    assert normalizer.normalize_word("ako") == "ako"
    assert normalizer.normalize_word("kumain") == "kumain"
    assert normalizer.normalize_word("u") == "u"


def test_curated_whole_form_mappings_precede_automatic_ranking():
    normalizer = FilipinoNormalizer()
    expected = {
        "yarn": "yan",
        "dehins": "hindi",
        "matsala": "salamat",
        "sakalam": "malakas",
        "naol": "sana all",
        "werpa": "power",
        "yorme": "mayor",
        "langya": "walang hiya",
    }

    assert {source: normalizer.normalize_word(source) for source in expected} == expected
    assert normalizer.normalize_word("Yarn") == "Yan"
    assert normalizer.normalize_word("YARN") == "YAN"
    assert normalizer.normalize_word("petmalu") == "malupit"
    assert normalizer.explain_word("naol") == {
        "input": "naol",
        "strategy": "curated_rule",
        "source_id": "0158",
        "output": "sana all",
        "curated_rule_status": "matched",
    }


def test_curated_phrase_punctuation_apostrophes_and_offsets_are_preserved():
    normalizer = FilipinoNormalizer()
    source = 'AQ, naol! "WER NA U?" mamsir, balakajan.'
    normalized, changes = normalizer.normalize_text(source)

    assert normalized == 'AKO, sana all! "NASAAN KA NA?" ma\'am at sir, bahala ka d\'yan.'
    assert [
        (item.word, item.suggestion, item.strategy, item.source_id)
        for item in changes
    ] == [
        ("AQ", "AKO", "ngram_dld", None),
        ("naol", "sana all", "curated_rule", "0158"),
        ("WER NA U?", "NASAAN KA NA?", "curated_rule", "0216"),
        ("mamsir", "ma'am at sir", "curated_rule", "0190"),
        ("balakajan", "bahala ka d'yan", "curated_rule", "0129"),
    ]
    for change in changes:
        assert source[change.start : change.end] == change.word

    one_to_many_source = "x Naol! y"
    one_to_many_output, one_to_many_changes = normalizer.normalize_text(one_to_many_source)
    assert one_to_many_output == "x Sana all! y"
    assert [
        (item.word, item.suggestion, item.start, item.end)
        for item in one_to_many_changes
    ] == [("Naol", "Sana all", 2, 6)]


def test_curated_matching_requires_a_complete_form_and_prefers_longest_phrase():
    normalizer = FilipinoNormalizer()
    automatic_only = FilipinoNormalizer(use_curated_mappings=False)

    assert normalizer.normalize_word("char") == "joke"
    assert normalizer.explain_word("charisma")["strategy"] == "ngram_dld"
    assert normalizer.normalize_text("charisma") == automatic_only.normalize_text("charisma")
    for larger_form in ("xchar", "char's", "char-like"):
        actual = normalizer.normalize_text(larger_form)
        expected = automatic_only.normalize_text(larger_form)
        assert actual == expected
        _output, changes = actual
        assert all(change.strategy != "curated_rule" for change in changes)
    _line_broken, line_broken_changes = normalizer.normalize_text("wer\nna u?")
    assert all(change.strategy != "curated_rule" for change in line_broken_changes)

    normalizer.curated_mappings = {
        "wer": {"normalized": "where", "source_id": "short"},
        "wer na u?": {"normalized": "long phrase", "source_id": "long"},
    }
    normalizer._curated_pattern = normalizer._compile_curated_pattern(normalizer.curated_mappings)
    result, changes = normalizer.normalize_text("(wer na u?)")

    assert result == "(long phrase)"
    assert [(change.word, change.source_id) for change in changes] == [("wer na u?", "long")]


def test_automatic_only_ablation_retains_frozen_prechange_outputs():
    normalizer = FilipinoNormalizer(use_curated_mappings=False)
    words = {
        "aq": "ako",
        "pra": "para",
        "kc": "ka",
        "ndi": "ndi",
        "skul": "skul",
        "tlga": "talaga",
        "nmn": "naman",
        "kse": "kase",
        "musta": "musta",
        "nandun": "nandun",
        "aqi": "aqi",
        "prang": "parang",
        "iPhone": "iPhone",
        "McDo": "McDo",
        "aKo": "aKo",
        "ako": "ako",
        "kumain": "kumain",
        "bahay": "bahay",
        "maganda": "maganda",
        "qzxv": "qzxv",
        "QZXV": "QZXV",
        "xyzzy": "xyzzy",
    }

    assert {source: normalizer.normalize_word(source) for source in words} == words
    source = "AQ, pra! KC? ndi; skul. qzxv https://example.com/aq"
    normalized, changes = normalizer.normalize_text(source)
    assert normalized == "AKO, para! KA? ndi; skul. qzxv https://example.com/aq"
    assert [
        (item.word, item.suggestion, item.start, item.end, item.strategy)
        for item in changes
    ] == [
        ("AQ", "AKO", 0, 2, "ngram_dld"),
        ("pra", "para", 4, 7, "ngram_dld"),
        ("KC", "KA", 9, 11, "ngram_dld"),
    ]
    assert normalizer.explain_word("kc")["strategy"] == "ngram_dld"
    assert all(
        normalizer.explain_word(source)["strategy"] == "ngram_dld"
        for source in ("aq", "pra", "kc", "ndi", "skul")
    )
    assert FilipinoNormalizer().normalize_word("aq") == normalizer.normalize_word("aq")


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
    with pytest.warns(RuntimeWarning, match="gora"):
        first = build_artifact_bytes(DEFAULT_PAIRS, DEFAULT_BASE_VOCABULARY)
    with pytest.warns(RuntimeWarning, match="gora"):
        second = build_artifact_bytes(DEFAULT_PAIRS, DEFAULT_BASE_VOCABULARY)

    assert first == second
    metadata = json.loads(first["metadata.json"])
    assert metadata["training_data"]["pair_count"] == 303
    assert metadata["algorithm_source"]["base_vocabulary_member"].endswith("with_info.txt")
    assert "test_words.csv" not in metadata["training_data"]["path"]
    assert first["rules.json"] == Path("artifacts/normalizer/rules.json").read_bytes()
    assert first["vocabulary.txt"] == Path("artifacts/normalizer/vocabulary.txt").read_bytes()
    assert metadata["curated_source"]["source_row_count"] == 1000
    assert metadata["curated_source"]["populated_mapping_row_count"] == 576
    assert metadata["curated_source"]["id_only_row_count"] == 424
    assert metadata["curated_source"]["slang_row_count"] == 100
    assert metadata["curated_source"]["unique_slang_input_count"] == 98
    assert metadata["curated_source"]["accepted_mapping_count"] == 97
    assert metadata["curated_source"]["duplicate_mapping_count"] == 1
    assert metadata["curated_source"]["ambiguous_mapping_count"] == 1


def test_curated_builder_selects_only_slang_and_reports_duplicates_and_conflicts(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "ID,Informal,Normalized,Category\n"
        "0001,petmalu,malupit, slang \n"
        "0002,petmalu,malupit,SLANG\n"
        "0003,gora,tara,slang\n"
        "0004,gora,lumarga,slang\n"
        "0005,word,normalized,abbreviation\n"
        '0006,combo,normalized,"abbreviation, spelling variation"\n',
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="gora"):
        resource, stats = build_curated_mappings(source)
    payload = json.loads(resource)

    assert set(payload["mappings"]) == {"petmalu"}
    assert payload["mappings"]["petmalu"]["source_id"] == "0001"
    assert payload["mappings"]["petmalu"]["source_ids"] == ["0001", "0002"]
    assert payload["ambiguous"]["gora"] == [
        {"normalized": "lumarga", "source_ids": ["0004"]},
        {"normalized": "tara", "source_ids": ["0003"]},
    ]
    assert stats["source_row_count"] == 6
    assert stats["slang_row_count"] == 4
    assert stats["unique_slang_input_count"] == 2
    assert stats["accepted_mapping_count"] == 1
    assert stats["duplicate_source_row_count"] == 1
    assert stats["ambiguous_mapping_count"] == 1
    assert "word" not in payload["mappings"]
    assert "combo" not in payload["mappings"]


def test_configured_artifact_paths_load_colocated_metadata_and_fail_closed(tmp_path, monkeypatch):
    default_artifacts = FilipinoNormalizer()._resource_signature[0].parent
    staged = tmp_path / "configured-normalizer"
    staged.mkdir()
    rules_path = staged / "custom-rules.json"
    vocabulary_path = staged / "custom-vocabulary.txt"
    for source, destination in (
        (default_artifacts / "rules.json", rules_path),
        (default_artifacts / "vocabulary.txt", vocabulary_path),
        (default_artifacts / "curated_mappings.json", staged / "curated_mappings.json"),
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
    for filename in ("rules.json", "vocabulary.txt", "curated_mappings.json", "metadata.json"):
        shutil.copyfile(default_artifacts / filename, other / filename)
    with pytest.raises(RuntimeError, match="different resource paths"):
        normalizer_service.get_normalizer(
            rules_path=other / "rules.json",
            vocabulary_path=other / "vocabulary.txt",
        )

    curated_path = staged / "curated_mappings.json"
    curated_bytes = curated_path.read_bytes()
    curated_path.unlink()
    with pytest.raises(FileNotFoundError, match="artifacts are incomplete"):
        FilipinoNormalizer(rules_path=rules_path, vocabulary_path=vocabulary_path)
    curated_path.write_bytes(curated_bytes + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FilipinoNormalizer(rules_path=rules_path, vocabulary_path=vocabulary_path)
    curated_path.write_bytes(curated_bytes)

    vocabulary_path.write_bytes(vocabulary_path.read_bytes() + b"corrupt\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FilipinoNormalizer(rules_path=rules_path, vocabulary_path=vocabulary_path)
