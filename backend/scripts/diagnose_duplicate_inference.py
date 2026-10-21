"""Build and run a bounded, provenance-pinned duplicate-word inference stress set.

Preparation reads only the production Stage-3 DEV parquet.  Inference uses the
normal FILLY GECService, records its P1..P5 trace, and observes first-subword
logits without changing the decoder.  No final-test data is loaded here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
FILLY_ROOT = BACKEND_ROOT.parent
WORKSPACE_ROOT = FILLY_ROOT.parent
DEFAULT_DEV = (
    WORKSPACE_ROOT
    / "Grammar-Error-Correction"
    / "defense_prereqs"
    / "gec_cpu"
    / "production-data"
    / "dev"
    / "dataset2_stage3.parquet"
)
DEFAULT_REPORTS = BACKEND_ROOT / "reports"
EXPECTED_DEV_SHA256 = "5a20d0e81c60791f537207635e74d89b1a57c49e371d65d67ab141ab19ef353e"
EXPECTED_CHECKPOINT_SHA256 = "4928c916cc27e9a04d53263e50ced5473c4cbd17317fc23a2cde10aff87e4a09"
EXPECTED_TRAIN_SHA256 = "53113e24278fd0bbaff66243c2308cf1e1b12741fee02aba8199718be6bc2ed3"
DEFAULT_SEED = 42

FUNCTION_WORDS = frozenset(
    "ang ng sa na ay at mga si ni nina kina kay para upang kung kapag dahil "
    "pero ngunit o or na rin din pa raw daw ito iyon iyan dito diyan doon "
    "rito roon siya sila kami tayo kayo nila niya namin natin inyong kanilang "
    "with the a an of to in for on from by and is are was were be been being "
    "that this these those as than then but if while when where who whom whose "
    "which what not no yes its their his her our your my me him them we they "
    "he she it you i"
    .split()
)
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*(?:[-‐‑][\w]+(?:['’][\w]+)*)*", re.UNICODE)
TERMINAL_PUNCTUATION = frozenset(".!?…")
PUNCTUATION_CONTEXT = frozenset(",;:—–")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank(seed: int, stratum: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{stratum}\0{key}".encode("utf-8")).hexdigest()


def _canonical_alignment(source: str, target: str) -> Any:
    source_dir = WORKSPACE_ROOT / "Grammar-Error-Correction" / "src"
    if not source_dir.is_dir():
        raise RuntimeError(f"canonical GEC alignment source is missing: {source_dir}")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from filly_gec.tagging.align import align_pair
    return align_pair(source, target)


def _require_dev_source(path: Path, rows: list[dict[str, Any]], actual_sha: str) -> None:
    lowered = str(path).casefold().replace("\\", "/")
    if "synthetic_test" in lowered or Path(path).name.casefold() == "final.parquet":
        raise ValueError("refusing synthetic_test/final-test input; use the Stage-3 DEV parquet")
    if actual_sha != EXPECTED_DEV_SHA256:
        raise ValueError(
            f"DEV parquet SHA-256 mismatch: expected {EXPECTED_DEV_SHA256}, got {actual_sha}"
        )
    if len(rows) != 50_400:
        raise ValueError(f"Stage-3 DEV row count mismatch: expected 50400, got {len(rows)}")
    if any(row.get("split") != "dev" or row.get("dataset_stage") != "dataset2_stage3" for row in rows):
        raise ValueError("source parquet contains rows outside split=dev, dataset_stage=dataset2_stage3")


def _text_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in WORD_RE.finditer(text)]


def _has_adjacent_repeat(text: str) -> bool:
    words = _text_tokens(text)
    return any(a[0].casefold() == b[0].casefold() for a, b in zip(words, words[1:]))


def _clean_identity(row: Mapping[str, Any]) -> bool:
    return row.get("split") == "dev" and row.get("dataset_stage") == "dataset2_stage3" \
        and row.get("is_errorful") is False and row.get("source_text") == row.get("target_text")


def _row_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "pair_id", "clean_id", "split", "dataset_stage", "source_corpus", "publisher",
        "source_doc_id", "sqlite_table", "sqlite_rowid", "seed", "generator_version",
        "candidate_builder_version", "dataset_builder_version", "config_hash", "quality_flags",
    )
    return {key: row.get(key) for key in keys}


def discover_corruption_candidates(
    rows: Iterable[Mapping[str, Any]], *, seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Enumerate clean-DEV token loci for deterministic adjacent-word injection."""
    # Import the actual FILLY tokenizer/segmenter so offsets and boundaries are
    # identical to inference, even when this preparation command runs in GEC's
    # PyArrow environment.
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from app.services.gec_checkpoint.segmentation import segment_sentences
    from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets

    candidates: dict[str, list[dict[str, Any]]] = {
        "interior_function": [], "interior_content": [],
        "initial_function": [], "initial_content": [],
        "punctuation_function": [], "final_content": [],
    }
    for row in rows:
        if not _clean_identity(row):
            continue
        target = row["target_text"]
        if not isinstance(target, str) or not target.strip() or _has_adjacent_repeat(target):
            continue
        lexical = _text_tokens(target)
        if not lexical:
            continue
        tokenized = tokenize_with_offsets(target)
        sentence_lexical: list[list[int]] = []
        layout = segment_sentences(target)
        for sentence in layout.sentences:
            indexes = [
                index for index, span in enumerate(tokenized.spans)
                if span.start >= sentence.start and span.end <= sentence.end
                and any(char.isalnum() for char in span.text)
            ]
            if indexes:
                sentence_lexical.append(indexes)
        sentence_by_index = {
            token_index: (sentence_index, indexes)
            for sentence_index, indexes in enumerate(sentence_lexical)
            for token_index in indexes
        }
        lexical_spans = [
            span for span in tokenized.spans if any(char.isalnum() for char in span.text)
        ]
        # Long/short classification uses correction-token count for the source
        # sentence, not character count. Medium sentences are not selected.
        for token_index, span in enumerate(tokenized.spans):
            if token_index not in sentence_by_index or not any(char.isalpha() for char in span.text):
                continue
            sentence_index, indexes = sentence_by_index[token_index]
            sentence_len = len(indexes)
            if sentence_len <= 15:
                length_bucket = "short"
            elif sentence_len >= 25:
                length_bucket = "long"
            else:
                continue
            token = span.text
            token_class = "function" if token.casefold() in FUNCTION_WORDS else "content"
            local_position = indexes.index(token_index)
            first_index, last_index = indexes[0], indexes[-1]
            next_span = tokenized.spans[token_index + 1] if token_index + 1 < len(tokenized.spans) else None
            prev_span = tokenized.spans[token_index - 1] if token_index else None
            punct_neighbor = bool(
                (next_span and next_span.text in PUNCTUATION_CONTEXT)
                or (prev_span and prev_span.text in PUNCTUATION_CONTEXT)
            )
            terminal_after = bool(
                next_span and next_span.text in TERMINAL_PUNCTUATION
            ) or (token_index == len(tokenized.spans) - 1 and target.rstrip()[-1:] in TERMINAL_PUNCTUATION)

            group_names: list[str] = []
            if token_index == first_index:
                group_names.append(f"initial_{token_class}")
            elif token_index == last_index and terminal_after and token_class == "content":
                group_names.append("final_content")
            elif punct_neighbor and token_class == "function":
                group_names.append("punctuation_function")
            elif first_index < token_index < last_index:
                group_names.append(f"interior_{token_class}")

            for group in group_names:
                if group not in candidates:
                    continue
                insert_at = span.end
                source = target[:insert_at] + " " + token + target[insert_at:]
                duplicate_span = {"start": insert_at + 1, "end": insert_at + 1 + len(token), "text": token}
                candidate_key = f"{row.get('pair_id')}:{token_index}:{group}"
                candidates[group].append({
                    "candidate_key": candidate_key,
                    "rank": _rank(seed, group, candidate_key),
                    "group": group,
                    "token_class": token_class,
                    "length_bucket": length_bucket,
                    "sentence_token_count": sentence_len,
                    "target_token_index": token_index,
                    "duplicate_source_span": duplicate_span,
                    "original_source_span": {"start": span.start, "end": span.end, "text": span.text},
                    "target_text": target,
                    "source_text": source,
                    "case_context": "initial_capital" if token[:1].isupper() else "preserved_case",
                    "punctuation_context": "adjacent_to_punctuation" if punct_neighbor or terminal_after else "none",
                    "provenance": _row_provenance(row),
                    "source_row_sha256": canonical_sha256(_row_provenance(row) | {"source": row.get("source_text"), "target": target}),
                })
    return candidates


def _select_candidates(
    candidates: Mapping[str, list[dict[str, Any]]], *, seed: int = DEFAULT_SEED,
    quotas: Mapping[str, Mapping[str, int]] | None = None,
    rejection_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if quotas is None:
        quotas = {
            "interior_function": {"short": 10, "long": 10},
            "interior_content": {"short": 10, "long": 10},
            "initial_function": {"short": 7, "long": 8},
            "initial_content": {"short": 7, "long": 8},
            "punctuation_function": {"short": 8, "long": 7},
            "final_content": {"short": 8, "long": 7},
        }
    chosen: list[dict[str, Any]] = []
    used_clean_ids: set[str] = set()
    insufficient: list[str] = []
    rejected: list[dict[str, str]] = []
    for group, length_quotas in quotas.items():
        for length_bucket, count in length_quotas.items():
            pool = [
                candidate for candidate in candidates.get(group, [])
                if candidate["length_bucket"] == length_bucket
                and candidate["provenance"].get("clean_id") not in used_clean_ids
            ]
            pool.sort(key=lambda item: (_rank(seed, group + ":" + length_bucket, item["candidate_key"]), item["candidate_key"]))
            selected: list[dict[str, Any]] = []
            for candidate in pool:
                clean_id = str(candidate["provenance"].get("clean_id"))
                if clean_id in used_clean_ids:
                    continue
                try:
                    aligned = _canonical_alignment(candidate["source_text"], candidate["target_text"])
                except Exception as exc:
                    rejection = {"candidate_key": candidate["candidate_key"], "stratum": group,
                                 "selection_rank": _rank(seed, group + ":" + length_bucket, candidate["candidate_key"]),
                                 "clean_id": candidate["provenance"].get("clean_id"),
                                 "source_row_sha256": candidate["source_row_sha256"],
                                 "reason": f"{type(exc).__name__}: {exc}"}
                    rejected.append(rejection)
                    continue
                if aligned.reconstructed_text != candidate["target_text"]:
                    rejected.append({
                        "candidate_key": candidate["candidate_key"], "stratum": group,
                        "selection_rank": _rank(seed, group + ":" + length_bucket, candidate["candidate_key"]),
                        "clean_id": candidate["provenance"].get("clean_id"),
                        "source_row_sha256": candidate["source_row_sha256"],
                        "reason": "canonical_alignment_replay_mismatch",
                    })
                    continue
                candidate = dict(candidate)
                candidate["canonical_alignment"] = {
                    "source_tokens": list(aligned.source_tokens),
                    "target_tokens": list(aligned.target_tokens),
                    "labels": list(aligned.labels),
                    "reconstructed_text": aligned.reconstructed_text,
                }
                candidate["selection_rank"] = _rank(
                    seed, group + ":" + length_bucket, candidate["candidate_key"]
                )
                selected.append(candidate)
                used_clean_ids.add(clean_id)
                if len(selected) == count:
                    break
            if len(selected) < count:
                insufficient.append(f"{group}/{length_bucket}: need {count}, have {len(selected)}")
            for item in selected:
                used_clean_ids.add(str(item["provenance"].get("clean_id")))
                chosen.append(item)
    if insufficient:
        raise ValueError("insufficient clean DEV candidates: " + "; ".join(insufficient))
    for item in chosen:
        item["alignment_rejections_seen"] = len(rejected)
    if rejection_log is not None:
        rejection_log.extend(rejected)
    return chosen


def _control_cases(rows: Iterable[Mapping[str, Any]], *, seed: int, used_clean_ids: set[str]) -> list[dict[str, Any]]:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets

    pools: dict[str, list[dict[str, Any]]] = {"identity_repeat": [], "identity_plain": []}
    for row in rows:
        if not _clean_identity(row) or str(row.get("clean_id")) in used_clean_ids:
            continue
        text = row["target_text"]
        tokenized = tokenize_with_offsets(text)
        lexical = [
            (index, span) for index, span in enumerate(tokenized.spans)
            if any(char.isalnum() for char in span.text)
        ]
        repeated_spans: list[dict[str, Any]] = []
        for (left_index, left), (right_index, right) in zip(lexical, lexical[1:]):
            if left.text.casefold() == right.text.casefold():
                repeated_spans.append({"start": right.start, "end": right.end, "text": right.text,
                                       "left_start": left.start, "left_end": left.end,
                                       "token_index": right_index})
        kind = "identity_repeat" if repeated_spans else "identity_plain"
        key = str(row.get("pair_id"))
        pools[kind].append({
            "candidate_key": key,
            "rank": _rank(seed, kind, key),
            "kind": kind,
            "source_text": text,
            "target_text": text,
            "focus_spans": repeated_spans,
            "provenance": _row_provenance(row),
            "source_row_sha256": canonical_sha256(_row_provenance(row) | {"source": text, "target": text}),
        })
    controls: list[dict[str, Any]] = []
    for kind, quota in (("identity_repeat", 10), ("identity_plain", 10)):
        pools[kind].sort(key=lambda item: (_rank(seed, kind, item["candidate_key"]), item["candidate_key"]))
        if len(pools[kind]) < quota:
            raise ValueError(f"need {quota} {kind} controls, found {len(pools[kind])}")
        controls.extend(pools[kind][:quota])
    return controls


def prepare_cases(source_path: Path, output_path: Path, manifest_path: Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("case preparation requires PyArrow; use the GEC environment") from exc
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = sha256_file(source_path)
    rows = parquet.read_table(source_path).to_pylist()
    _require_dev_source(source_path, rows, source_sha)
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets
    candidates = discover_corruption_candidates(rows, seed=seed)
    quotas = {
        "interior_function": {"short": 10, "long": 10},
        "interior_content": {"short": 10, "long": 10},
        "initial_function": {"short": 7, "long": 8},
        "initial_content": {"short": 7, "long": 8},
        "punctuation_function": {"short": 8, "long": 7},
        "final_content": {"short": 8, "long": 7},
    }
    alignment_rejections: list[dict[str, Any]] = []
    corruptions = _select_candidates(candidates, seed=seed, quotas=quotas,
                                     rejection_log=alignment_rejections)
    used_clean_ids = {str(item["provenance"].get("clean_id")) for item in corruptions}
    controls = _control_cases(rows, seed=seed, used_clean_ids=used_clean_ids)
    cases: list[dict[str, Any]] = []
    for index, candidate in enumerate(corruptions, 1):
        case_id = hashlib.sha256(candidate["candidate_key"].encode("utf-8")).hexdigest()[:20]
        alignment = candidate["canonical_alignment"]
        focus_spans = []
        for role, span in (("original_copy", candidate["original_source_span"]),
                           ("inserted_copy", candidate["duplicate_source_span"])):
            token_index = next(
                (idx for idx, source_span in enumerate(tokenize_with_offsets(candidate["source_text"]).spans)
                 if source_span.start == span["start"] and source_span.end == span["end"]),
                None,
            )
            if token_index is None or token_index >= len(alignment["labels"]):
                raise ValueError(f"could not align focus span in candidate {candidate['candidate_key']}")
            focus_spans.append({**span, "role": role, "source_token_index": token_index,
                                "canonical_gold_label": alignment["labels"][token_index]})
        cases.append({
            "case_id": case_id,
            "case_type": "new_duplicate_corruption",
            "stratum": candidate["group"],
            "target_class": candidate["token_class"],
            "length_bucket": candidate["length_bucket"],
            "sentence_token_count": candidate["sentence_token_count"],
            "source_text": candidate["source_text"],
            "target_text": candidate["target_text"],
            "expected_operation": "$DELETE",
            "duplicate_source_span": candidate["duplicate_source_span"],
            "focus_spans": focus_spans,
            "target_token_index": candidate["target_token_index"],
            "canonical_alignment": alignment,
            "case_context": candidate["case_context"],
            "punctuation_context": candidate["punctuation_context"],
            "provenance": candidate["provenance"],
            "source_row_sha256": candidate["source_row_sha256"],
            "selection_seed": seed,
            "selection_rank": candidate["selection_rank"],
            "candidate_level_rank": candidate["rank"],
            "corruption_ordinal": index,
        })
    for index, control in enumerate(controls, 1):
        case_id = hashlib.sha256((control["kind"] + ":" + control["candidate_key"]).encode("utf-8")).hexdigest()[:20]
        cases.append({
            "case_id": case_id,
            "case_type": control["kind"],
            "stratum": control["kind"],
            "target_class": None,
            "length_bucket": None,
            "sentence_token_count": len(_text_tokens(control["target_text"])),
            "source_text": control["source_text"],
            "target_text": control["target_text"],
            "expected_operation": None,
            "duplicate_source_span": None,
            "focus_spans": control["focus_spans"],
            "provenance": control["provenance"],
            "source_row_sha256": control["source_row_sha256"],
            "selection_seed": seed,
            "selection_rank": control["rank"],
            "control_ordinal": index,
        })
    if len({case["case_id"] for case in cases}) != len(cases):
        raise AssertionError("case ID collision")
    if len(corruptions) != 100 or len(controls) != 20:
        raise AssertionError("expected exactly 100 new corruptions and 20 identity controls")
    class_counts = Counter(item["token_class"] for item in corruptions)
    length_counts = Counter(item["length_bucket"] for item in corruptions)
    stratum_counts = Counter(item["group"] for item in corruptions)
    if class_counts != {"function": 50, "content": 50}:
        raise AssertionError(f"function/content balance failed: {class_counts}")
    if length_counts != {"short": 50, "long": 50}:
        raise AssertionError(f"short/long balance failed: {length_counts}")
    manifest = {
        "schema_version": 1,
        "status": "PREPARED",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "source": {
            "path_from_workspace": source_path.relative_to(WORKSPACE_ROOT).as_posix(),
            "sha256": source_sha,
            "rows": len(rows),
            "split": "dev",
            "dataset_stage": "dataset2_stage3",
            "checkpoint_train_overlap_note": "Cases are generated from clean rows in checkpoint Stage-3 DEV; no synthetic_test rows are accessed.",
        },
        "sampling": {
            "algorithm": "SHA256(seed NUL stratum NUL candidate_key), ascending rank, first eligible row per clean_id",
            "synthetic_corruptions": 100,
            "identity_controls": 20,
            "function_word_targets": class_counts["function"],
            "content_word_targets": class_counts["content"],
            "short_sentence_cases_le_15_correction_tokens": length_counts["short"],
            "long_sentence_cases_ge_25_correction_tokens": length_counts["long"],
            "corruption_strata": dict(sorted(stratum_counts.items())),
            "control_strata": dict(Counter(item["case_type"] for item in cases if item["case_type"].startswith("identity_"))),
            "target_text_source": "source_text == target_text from Stage-3 DEV identity rows",
            "operation": "insert one exact adjacent copy of an existing lexical token, preserving all original punctuation and source gaps",
            "canonical_alignment": "Before quota selection, retain only cases where Grammar-Error-Correction/src/filly_gec/tagging/align.py::align_pair exactly replays the generated target; replacement selection is SHA-ranked and independent of model output.",
            "canonical_alignment_rejections": corruptions[0].get("alignment_rejections_seen", 0),
            "canonical_alignment_rejection_log_sha256": canonical_sha256(alignment_rejections),
        },
        "checkpoint_reference": {"path": "Filly/best.pt", "sha256": EXPECTED_CHECKPOINT_SHA256},
        "case_file": output_path.name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_path = manifest_path.with_name("duplicate_inference_rejections.json")
    rejection_path.write_text(json.dumps(alignment_rejections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["sampling"]["canonical_alignment_rejection_log"] = rejection_path.name
    manifest["sampling"]["canonical_alignment_rejection_log_file_sha256"] = sha256_file(rejection_path)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["cases_sha256"] = sha256_file(output_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_case_provenance(source_path: Path, cases_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Write selection-rank and alignment-rejection audits without changing cases."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("case provenance audit requires PyArrow; use the GEC environment") from exc
    source_path = source_path.expanduser().resolve()
    cases_path = cases_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases_sha = sha256_file(cases_path)
    if manifest.get("cases_sha256") != cases_sha:
        raise ValueError("case file changed since preparation; refusing to audit altered cases")
    source_sha = sha256_file(source_path)
    rows = parquet.read_table(source_path).to_pylist()
    _require_dev_source(source_path, rows, source_sha)
    seed = int(manifest["seed"])
    candidates = discover_corruption_candidates(rows, seed=seed)
    quotas = {
        "interior_function": {"short": 10, "long": 10},
        "interior_content": {"short": 10, "long": 10},
        "initial_function": {"short": 7, "long": 8},
        "initial_content": {"short": 7, "long": 8},
        "punctuation_function": {"short": 8, "long": 7},
        "final_content": {"short": 8, "long": 7},
    }
    rejected: list[dict[str, Any]] = []
    selected = _select_candidates(candidates, seed=seed, quotas=quotas, rejection_log=rejected)
    selected_ids = [hashlib.sha256(item["candidate_key"].encode("utf-8")).hexdigest()[:20]
                    for item in selected]
    case_ids = [item["case_id"] for item in cases if item["case_type"] == "new_duplicate_corruption"]
    if selected_ids != case_ids:
        raise ValueError("recomputed canonical selection does not match the sealed case file")
    selected_by_id = dict(zip(selected_ids, selected))
    rank_rows = []
    for case in cases:
        if case["case_type"] == "new_duplicate_corruption":
            candidate = selected_by_id[case["case_id"]]
            candidate_key = candidate["candidate_key"]
            actual_rank = candidate["selection_rank"]
            rank_semantics = "sealed-case rank uses group-only candidate rank; actual quota rank follows"
        else:
            candidate_key = str(case["provenance"].get("pair_id"))
            actual_rank = _rank(seed, case["case_type"], candidate_key)
            rank_semantics = "identity-control stratum rank"
        rank_rows.append({
            "case_id": case["case_id"], "case_type": case["case_type"],
            "candidate_key": candidate_key,
            "sealed_case_json_rank": case.get("selection_rank"),
            "sealed_case_json_rank_semantics": rank_semantics,
            "actual_quota_selection_rank": actual_rank,
            "rank_matches_actual": case.get("selection_rank") == actual_rank,
        })
    reports_dir = manifest_path.parent
    rejection_path = reports_dir / "duplicate_inference_rejections.json"
    selection_path = reports_dir / "duplicate_inference_selection_ranks.json"
    rejection_path.write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rank_payload = {
        "schema_version": 1, "seed": seed, "cases_sha256": cases_sha,
        "selection_algorithm": "ascending SHA256(seed NUL (stratum:length_bucket) NUL candidate_key), first eligible canonical-replay candidate per clean_id",
        "rows": rank_rows,
    }
    selection_path.write_text(json.dumps(rank_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sampling = manifest.setdefault("sampling", {})
    sampling["canonical_alignment_rejections"] = len(rejected)
    sampling["canonical_alignment_rejection_log"] = rejection_path.name
    sampling["canonical_alignment_rejection_log_file_sha256"] = sha256_file(rejection_path)
    sampling["canonical_alignment_rejection_log_sha256"] = canonical_sha256(rejected)
    sampling["selection_rank_audit"] = selection_path.name
    sampling["selection_rank_audit_sha256"] = sha256_file(selection_path)
    sampling["selection_rank_note"] = "The sealed case file's selection_rank records the candidate-level rank. Actual quota-selection ranks are stored in the sidecar. The case choices and case-file SHA-256 are unchanged."
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "case_count": len(rank_rows), "rejection_count": len(rejected),
        "rejection_file": str(rejection_path), "selection_rank_file": str(selection_path),
        "cases_sha256": cases_sha,
        "rank_matches": sum(row["rank_matches_actual"] for row in rank_rows),
    }


class _TokenizerTap:
    """Transparent tokenizer proxy that exposes word_ids to the logit hook."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.last_encoded: Any = None
        self.last_words: list[list[str]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        words = args[0] if args else kwargs.get("text")
        encoded = self.base(*args, **kwargs)
        self.last_encoded = encoded
        self.last_words = [list(row) for row in words]
        return encoded

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def _capture_focus_topk(
    service: Any, tap: _TokenizerTap, cases: list[dict[str, Any]], *, top_k: int = 5,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import torch
    from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets

    active: dict[str, Any] = {"iteration": 0, "case_id": None, "focus_indices": set(), "cursor": 0}
    captured: dict[str, list[dict[str, Any]]] = {}
    hook_facts = {"forward_calls": 0, "all_observed_inference_mode": True, "all_observed_eval_mode": True}
    keep_id = service.label_vocab["$KEEP"]
    delete_id = service.label_vocab["$DELETE"]
    id_to_label = {index: label for label, index in service.label_vocab.items()}

    original_parts = service._correct_sentence_parts
    def traced_parts(text: str, sentence_texts: tuple[str, ...], gaps: tuple[str, ...], *, iteration: int):
        active["iteration"] = iteration
        return original_parts(text, sentence_texts, gaps, iteration=iteration)
    service._correct_sentence_parts = traced_parts

    def forward_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hook_facts["forward_calls"] += 1
        hook_facts["all_observed_inference_mode"] &= torch.is_inference_mode_enabled()
        hook_facts["all_observed_eval_mode"] &= not service.model.training
        if active["iteration"] != 1 or active["case_id"] is None:
            return
        encoded = tap.last_encoded
        words = tap.last_words
        if encoded is None or len(words) != output.correction_logits.shape[0]:
            raise RuntimeError("tokenizer tap did not match the current model batch")
        raw_probs = torch.softmax(output.correction_logits.detach(), dim=-1)
        adjusted = raw_probs.clone()
        adjusted[..., keep_id] += 0.2
        detect_probs = torch.softmax(output.detection_logits.detach(), dim=-1)[..., 1]
        k = min(top_k, raw_probs.shape[-1])
        top_probs, top_ids = raw_probs.topk(k, dim=-1)
        for row_index, source_words in enumerate(words):
            word_ids = encoded.word_ids(row_index)
            first_piece: dict[int, int] = {}
            piece_counts: Counter[int] = Counter()
            for position, word_id in enumerate(word_ids):
                if word_id is not None:
                    piece_counts[word_id] += 1
                    first_piece.setdefault(word_id, position)
            if sorted(first_piece) != list(range(len(source_words))):
                raise RuntimeError("fast-tokenizer word IDs did not cover every correction token")
            for local_index, word in enumerate(source_words):
                global_index = active["cursor"]
                active["cursor"] += 1
                if global_index not in active["focus_indices"]:
                    continue
                position = first_piece[local_index]
                raw_argmax = int(raw_probs[row_index, position].argmax().item())
                adjusted_argmax = int(adjusted[row_index, position].argmax().item())
                detection_probability = float(detect_probs[row_index, position].item())
                top = [
                    {"label": id_to_label[int(label_id)], "probability": float(probability)}
                    for probability, label_id in zip(top_probs[row_index, position].tolist(), top_ids[row_index, position].tolist())
                ]
                delete_rank = next((index + 1 for index, item in enumerate(top) if item["label"] == "$DELETE"), None)
                captured.setdefault(active["case_id"], []).append({
                    "global_token_index": global_index,
                    "token": word,
                    "first_subword_position": position,
                    "subword_piece_count": piece_counts[local_index],
                    "top_k_raw_correction": top,
                    "delete_raw_rank": delete_rank,
                    "raw_delete_probability": float(raw_probs[row_index, position, delete_id].item()),
                    "raw_keep_probability": float(raw_probs[row_index, position, keep_id].item()),
                    "delete_over_keep_bonus_margin": float(
                        raw_probs[row_index, position, delete_id].item()
                        - raw_probs[row_index, position, keep_id].item() - 0.2
                    ),
                    "raw_argmax_label": id_to_label[raw_argmax],
                    "keep_adjusted_argmax_label": id_to_label[adjusted_argmax],
                    "detection_error_probability": detection_probability,
                    "detection_gate_passes": detection_probability >= 0.5,
                    "gate_blocked_delete": id_to_label[adjusted_argmax] == "$DELETE" and detection_probability < 0.5,
                    "keep_bonus_suppressed_raw_delete": id_to_label[raw_argmax] == "$DELETE" and id_to_label[adjusted_argmax] == "$KEEP",
                    "decoded_label": id_to_label[adjusted_argmax] if detection_probability >= 0.5 else "$KEEP",
                })

    handle = service.model.register_forward_hook(forward_hook)
    for case in cases:
        tokenized = tokenize_with_offsets(case["source_text"])
        focused: set[int] = set()
        for focus in case.get("focus_spans", []):
            for index, span in enumerate(tokenized.spans):
                if span.start == focus["start"] and span.end == focus["end"]:
                    focused.add(index)
        active.update({"iteration": 0, "case_id": case["case_id"], "focus_indices": focused, "cursor": 0})
        captured[case["case_id"]] = []
        # The caller executes the real five-pass service path once per case.
        case["_topk_focus_indices"] = sorted(focused)
    return captured, {"hook": hook_facts, "handle": handle, "active": active, "original_parts": original_parts}


def _pass_record(result: Any, index: int) -> dict[str, Any]:
    from app.services.gec_checkpoint.segmentation import segment_sentences
    return {
        "iteration": result.iteration,
        "input_text": result.input_text,
        "output_text": result.output_text,
        "labels": list(result.labels),
        "changes": [
            {"original": change.original, "correction": change.correction,
             "start": change.start, "end": change.end, "label": change.label,
             "confidence": change.confidence}
            for change in result.changes
        ],
        "model_invoked": result.model_invoked,
        "fresh_segment_count": len(segment_sentences(result.output_text).sentences),
        "fixed_initial_slot_count": index,
    }


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    from app.evaluation.engine import _load_edit_metrics

    final_output_edit_counts, _ = _load_edit_metrics()
    scorer_module = __import__(final_output_edit_counts.__module__, fromlist=["_prf"])
    prf = scorer_module._prf
    summary: dict[str, Any] = {"by_pass": [], "case_counts": dict(Counter(p["case_type"] for p in predictions))}
    pass_count = max((len(p.get("passes", [])) for p in predictions), default=0)
    for pass_index in range(pass_count):
        groups: dict[str, dict[str, int]] = {}
        all_cases = [p for p in predictions if len(p.get("passes", [])) > pass_index]
        for group in sorted({p["case_type"] for p in all_cases}):
            rows = [p for p in all_cases if p["case_type"] == group]
            exact = sum(row["passes"][pass_index]["output_text"] == row["target_text"] for row in rows)
            edit_counts = [
                final_output_edit_counts(row["source_text"], row["target_text"], row["passes"][pass_index]["output_text"])
                for row in rows
            ]
            tp = sum(item["tp"] for item in edit_counts)
            fp = sum(item["fp"] for item in edit_counts)
            fn = sum(item["fn"] for item in edit_counts)
            gold_edits = sum(item["gold_edit_count"] for item in edit_counts)
            residual_edits = sum(
                final_output_edit_counts(row["target_text"], row["passes"][pass_index]["output_text"],
                                         row["passes"][pass_index]["output_text"])["gold_edit_count"]
                for row in rows
            )
            scores = prf(tp, fp, fn)
            false_edit = sum(
                row["case_type"].startswith("identity_")
                and row["passes"][pass_index]["output_text"] != row["target_text"]
                for row in rows
            )
            focus_delete = 0
            if pass_index == 0:
                focus_delete = sum(
                    item.get("decoded_label") == "$DELETE"
                    for row in rows for item in row.get("topk_focus", [])
                )
            groups[group] = {
                "n": len(rows), "exact_matches": exact,
                "exact_match_rate": exact / len(rows) if rows else 0.0,
                "tp": tp, "fp": fp, "fn": fn, "gold_edits": gold_edits,
                "residual_edits": residual_edits,
                "precision": float(scores["precision"]), "recall": float(scores["recall"]),
                "f0_5": float(scores["f0_5"]),
                "err": (gold_edits - residual_edits) / gold_edits if gold_edits else None,
                "new_corruption_exact_repairs": exact if group == "new_duplicate_corruption" else None,
                "identity_false_edit_rows": false_edit,
                "p1_focus_positions_decoded_delete": focus_delete,
            }
        combined_rows = all_cases
        combined_counts = [
            final_output_edit_counts(row["source_text"], row["target_text"], row["passes"][pass_index]["output_text"])
            for row in combined_rows
        ]
        tp = sum(item["tp"] for item in combined_counts)
        fp = sum(item["fp"] for item in combined_counts)
        fn = sum(item["fn"] for item in combined_counts)
        gold_edits = sum(item["gold_edit_count"] for item in combined_counts)
        residual_edits = sum(
            final_output_edit_counts(row["target_text"], row["passes"][pass_index]["output_text"],
                                     row["passes"][pass_index]["output_text"])["gold_edit_count"]
            for row in combined_rows
        )
        scores = prf(tp, fp, fn)
        summary["by_pass"].append({
            "pass": pass_index + 1, "groups": groups,
            "new_corruption_exact_by_class": {
                target_class: {
                    "n": sum(row.get("target_class") == target_class for row in all_cases
                              if row["case_type"] == "new_duplicate_corruption"),
                    "exact_matches": sum(
                        row.get("target_class") == target_class
                        and row["passes"][pass_index]["output_text"] == row["target_text"]
                        for row in all_cases if row["case_type"] == "new_duplicate_corruption"
                    ),
                }
                for target_class in ("function", "content")
            },
            "overall_edit_metrics": {
                "n": len(combined_rows), "tp": tp, "fp": fp, "fn": fn,
                "gold_edits": gold_edits, "residual_edits": residual_edits,
                "precision": float(scores["precision"]), "recall": float(scores["recall"]),
                "f0_5": float(scores["f0_5"]),
                "err": (gold_edits - residual_edits) / gold_edits if gold_edits else None,
            },
        })
    return summary


def validate_prepared_cases(cases_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Bind run inputs to the pinned DEV preparation before loading weights."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    if (source.get("sha256") != EXPECTED_DEV_SHA256
            or source.get("split") != "dev"
            or source.get("dataset_stage") != "dataset2_stage3"
            or source.get("rows") != 50_400):
        raise ValueError("case manifest is not bound to the pinned Stage-3 DEV source")
    if manifest.get("checkpoint_reference", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("case manifest checkpoint hash differs from the selected checkpoint")
    if manifest.get("case_file") != cases_path.name or manifest.get("cases_sha256") != sha256_file(cases_path):
        raise ValueError("prepared case filename/hash mismatch")
    return manifest


def run_inference(
    cases_path: Path, predictions_path: Path, facts_path: Path, *, top_k: int = 5,
    limit_cases: int | None = None, manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path or cases_path.with_name("duplicate_inference_stress_manifest.json")
    validate_prepared_cases(cases_path, manifest_path)
    os.environ.setdefault("GECTOR_LOCAL_FILES_ONLY", "true")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from app.services.gec import GECService
    from app.services.gec_checkpoint.segmentation import segment_sentences
    from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit_cases is not None:
        if limit_cases < 1:
            raise ValueError("limit_cases must be positive")
        cases = cases[:limit_cases]
    elif len(cases) < 100:
        raise ValueError(f"need at least 100 cases, got {len(cases)}")
    for case in cases:
        provenance = case.get("provenance", {})
        if provenance.get("split") != "dev" or provenance.get("dataset_stage") != "dataset2_stage3":
            raise ValueError(f"case {case.get('case_id')} is not from Stage-3 DEV")
    service = GECService()
    if service.metadata.get("stage") != "stage3" or not service.metadata.get("selected_best"):
        raise RuntimeError("loaded checkpoint is not the selected Stage-3 best checkpoint")
    if service.iterations < 1:
        raise RuntimeError("service configured iterations must be positive")
    tap = _TokenizerTap(service.tokenizer)
    service.tokenizer = tap
    service.predictor.tokenizer = tap
    topk_rows: dict[str, list[dict[str, Any]]] = {}
    hook_facts: dict[str, Any] = {"forward_calls": 0, "all_observed_inference_mode": True, "all_observed_eval_mode": True}
    phase = {"iteration": 0, "case_id": None, "focus_indices": set(), "focus_info": {}, "cursor": 0}
    id_to_label = {index: label for label, index in service.label_vocab.items()}
    keep_id = service.label_vocab["$KEEP"]
    delete_id = service.label_vocab["$DELETE"]
    original_parts = service._correct_sentence_parts
    def traced_parts(text: str, sentence_texts: tuple[str, ...], gaps: tuple[str, ...], *, iteration: int):
        phase["iteration"] = iteration
        return original_parts(text, sentence_texts, gaps, iteration=iteration)
    service._correct_sentence_parts = traced_parts

    import torch
    def forward_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hook_facts["forward_calls"] += 1
        hook_facts["all_observed_inference_mode"] &= torch.is_inference_mode_enabled()
        hook_facts["all_observed_eval_mode"] &= not service.model.training
        if phase["iteration"] != 1 or phase["case_id"] is None:
            return
        encoded = tap.last_encoded
        words = tap.last_words
        if encoded is None or len(words) != output.correction_logits.shape[0]:
            raise RuntimeError("tokenizer tap did not match the current model batch")
        raw_probs = torch.softmax(output.correction_logits.detach(), dim=-1)
        adjusted = raw_probs.clone()
        adjusted[..., keep_id] += 0.2
        detect_probs = torch.softmax(output.detection_logits.detach(), dim=-1)[..., 1]
        k = min(top_k, raw_probs.shape[-1])
        top_probs, top_ids = raw_probs.topk(k, dim=-1)
        for row_index, source_words in enumerate(words):
            word_ids = encoded.word_ids(row_index)
            first_piece: dict[int, int] = {}
            piece_counts: Counter[int] = Counter()
            for position, word_id in enumerate(word_ids):
                if word_id is not None:
                    piece_counts[word_id] += 1
                    first_piece.setdefault(word_id, position)
            if sorted(first_piece) != list(range(len(source_words))):
                raise RuntimeError("fast-tokenizer word IDs did not cover every correction token")
            for local_index, word in enumerate(source_words):
                global_index = phase["cursor"]
                phase["cursor"] += 1
                if global_index not in phase["focus_indices"]:
                    continue
                focus_info = phase["focus_info"].get(global_index, {})
                position = first_piece[local_index]
                raw_argmax = int(raw_probs[row_index, position].argmax().item())
                adjusted_argmax = int(adjusted[row_index, position].argmax().item())
                detection_probability = float(detect_probs[row_index, position].item())
                top = [
                    {"label": id_to_label[int(label_id)], "probability": float(probability)}
                    for probability, label_id in zip(top_probs[row_index, position].tolist(), top_ids[row_index, position].tolist())
                ]
                delete_rank = next((index + 1 for index, item in enumerate(top) if item["label"] == "$DELETE"), None)
                topk_rows.setdefault(phase["case_id"], []).append({
                    "global_token_index": global_index, "token": word,
                    "focus_role": focus_info.get("role"),
                    "canonical_gold_label": focus_info.get("canonical_gold_label"),
                    "first_subword_position": position,
                    "subword_piece_count": piece_counts[local_index],
                    "top_k_raw_correction": top, "delete_raw_rank": delete_rank,
                    "raw_delete_probability": float(raw_probs[row_index, position, delete_id].item()),
                    "raw_keep_probability": float(raw_probs[row_index, position, keep_id].item()),
                    "delete_over_keep_bonus_margin": float(
                        raw_probs[row_index, position, delete_id].item()
                        - raw_probs[row_index, position, keep_id].item() - 0.2
                    ),
                    "raw_argmax_label": id_to_label[raw_argmax],
                    "keep_adjusted_argmax_label": id_to_label[adjusted_argmax],
                    "detection_error_probability": detection_probability,
                    "detection_gate_passes": detection_probability >= 0.5,
                    "gate_blocked_delete": id_to_label[adjusted_argmax] == "$DELETE" and detection_probability < 0.5,
                    "keep_bonus_suppressed_raw_delete": id_to_label[raw_argmax] == "$DELETE" and id_to_label[adjusted_argmax] == "$KEEP",
                    "decoded_label": id_to_label[adjusted_argmax] if detection_probability >= 0.5 else "$KEEP",
                })
    handle = service.model.register_forward_hook(forward_hook)
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    per_case_seconds: list[float] = []
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_stream = predictions_path.open("w", encoding="utf-8", newline="\n")
    try:
        for case_number, case in enumerate(cases, 1):
            case_started = time.perf_counter()
            tokenized = tokenize_with_offsets(case["source_text"])
            focused: set[int] = set()
            focus_info: dict[int, dict[str, Any]] = {}
            for focus in case.get("focus_spans", []):
                for index, span in enumerate(tokenized.spans):
                    if span.start == focus["start"] and span.end == focus["end"]:
                        focused.add(index)
                        focus_info[index] = focus
            phase.update({"iteration": 0, "case_id": case["case_id"], "focus_indices": focused,
                          "focus_info": focus_info, "cursor": 0})
            topk_rows[case["case_id"]] = []
            try:
                result = service.correct_iteratively(case["source_text"], iterations=5)
                pass_records: list[dict[str, Any]] = []
                initial_slots = len(segment_sentences(case["source_text"]).sentences)
                for pass_result in result.passes:
                    row = _pass_record(pass_result, initial_slots)
                    row["initial_sentence_slot_count"] = initial_slots
                    pass_records.append(row)
                p1_labels = result.passes[0].labels if result.passes else ()
                for item in topk_rows[case["case_id"]]:
                    index = item["global_token_index"]
                    item["decoded_label"] = p1_labels[index] if index < len(p1_labels) else item["decoded_label"]
                output = {
                    "case_id": case["case_id"], "case_type": case["case_type"],
                    "stratum": case.get("stratum"), "target_class": case.get("target_class"),
                    "length_bucket": case.get("length_bucket"),
                    "source_text": case["source_text"], "target_text": case["target_text"],
                    "passes": pass_records, "topk_focus": topk_rows[case["case_id"]],
                    "focus_label_p1": [
                        {"token_index": item["global_token_index"], "label": item["decoded_label"]}
                        for item in topk_rows[case["case_id"]]
                    ],
                    "initial_sentence_slot_count": initial_slots,
                    "provenance": case.get("provenance"),
                    "expected_operation": case.get("expected_operation"),
                    "duplicate_source_span": case.get("duplicate_source_span"),
                    "focus_spans": case.get("focus_spans", []),
                    "source_row_sha256": case.get("source_row_sha256"),
                }
            except Exception as exc:
                output = {
                    "case_id": case["case_id"], "case_type": case["case_type"],
                    "stratum": case.get("stratum"), "target_class": case.get("target_class"),
                    "length_bucket": case.get("length_bucket"),
                    "source_text": case["source_text"], "target_text": case["target_text"],
                    "passes": [], "topk_focus": topk_rows[case["case_id"]],
                    "provenance": case.get("provenance"), "error": f"{type(exc).__name__}: {exc}",
                }
            predictions.append(output)
            predictions_stream.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            predictions_stream.flush()
            per_case_seconds.append(time.perf_counter() - case_started)
            if case_number == 1 or case_number % 10 == 0 or case_number == len(cases):
                print(f"INFERENCE_PROGRESS completed={case_number}/{len(cases)} elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
    finally:
        predictions_stream.close()
        handle.remove()
        service._correct_sentence_parts = original_parts
    elapsed = time.perf_counter() - started
    checkpoint_path = service.checkpoint_path
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: expected pinned {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}")
    all_focus = [item for row in predictions for item in row.get("topk_focus", [])]
    positive_focus = [item for row in predictions if row["case_type"] == "new_duplicate_corruption" for item in row.get("topk_focus", [])]
    delete_keep_margins = [item["delete_over_keep_bonus_margin"] for item in positive_focus]
    result_summary = summarize_predictions(predictions)
    segmentation_by_pass = []
    for pass_index in range(5):
        changed_case_ids: list[str] = []
        added_slots = 0
        removed_slots = 0
        for row in predictions:
            if len(row.get("passes", [])) <= pass_index:
                continue
            pass_record = row["passes"][pass_index]
            initial = pass_record["initial_sentence_slot_count"]
            fresh = pass_record["fresh_segment_count"]
            if fresh != initial:
                changed_case_ids.append(row["case_id"])
            added_slots += max(0, fresh - initial)
            removed_slots += max(0, initial - fresh)
        segmentation_by_pass.append({
            "pass": pass_index + 1, "cases_with_fresh_segmentation_change": len(changed_case_ids),
            "additional_fresh_slots": added_slots, "fewer_fresh_slots": removed_slots,
            "case_ids_with_change": changed_case_ids,
        })
    from app.evaluation.engine import scorer_provenance
    facts = {
        "schema_version": 1,
        "status": "COMPLETED" if all(row.get("passes") for row in predictions) else "COMPLETED_WITH_CASE_ERRORS",
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases_sha256": sha256_file(cases_path),
        "case_count": len(cases),
        "limited_benchmark": limit_cases is not None,
        "source": {
            "path": "Grammar-Error-Correction/defense_prereqs/gec_cpu/production-data/dev/dataset2_stage3.parquet",
            "sha256": EXPECTED_DEV_SHA256,
            "split": "dev", "dataset_stage": "dataset2_stage3",
        },
        "case_errors": sum(not row.get("passes") for row in predictions),
        "active_configured_iterations": service.iterations,
        "active_configured_output_pass": min(service.iterations, 5),
        "diagnostic_requested_iterations": 5,
        "interpretation": "The active one-pass outcome is P1 from the same sequential pass trace; explicit P1..P5 avoids rerunning the already-identical first forward.",
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha, "size_bytes": checkpoint_path.stat().st_size,
                       "stage": service.metadata.get("stage"), "selected_best": service.metadata.get("selected_best"),
                       "run_id": service.metadata.get("run_id"), "format_version": service.metadata.get("checkpoint_format_version"),
                       "model_id": service.metadata.get("model_id"), "model_revision": service.metadata.get("model_revision"),
                       "label_vocab_size": service.metadata.get("label_vocab_size"), "label_vocab_sha256": service.metadata.get("label_vocab_sha256"),
                       "source_manifest_hash": service.metadata.get("source_manifest_hash"), "transformers_version": service.metadata.get("transformers_version")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": str(service.device),
                    "model_eval_mode_verified": hook_facts["all_observed_eval_mode"],
                    "torch_inference_mode_verified": hook_facts["all_observed_inference_mode"],
                    "forward_calls": hook_facts["forward_calls"], "top_k": top_k,
                    "keep_additional_confidence": 0.2, "minimum_detection_error_probability": 0.5,
                    "tokenizer_revision": service.metadata.get("model_revision"), "tokenizer_fast_prefix_space": True},
        "timing": {"inference_seconds": elapsed, "cases_per_second": len(cases) / elapsed if elapsed else None,
                   "mean_case_seconds": sum(per_case_seconds) / len(per_case_seconds) if per_case_seconds else None},
        "topk_diagnostics": {
            "focus_positions": len(all_focus),
            "positive_duplicate_focus_positions": len(positive_focus),
            "positive_raw_argmax_delete": sum(item["raw_argmax_label"] == "$DELETE" for item in positive_focus),
            "positive_keep_adjusted_argmax_delete": sum(item["keep_adjusted_argmax_label"] == "$DELETE" for item in positive_focus),
            "positive_keep_bonus_suppressed_raw_delete": sum(item["keep_bonus_suppressed_raw_delete"] for item in positive_focus),
            "positive_detection_gate_blocked_delete": sum(item["gate_blocked_delete"] for item in positive_focus),
            "positive_decoded_delete": sum(item["decoded_label"] == "$DELETE" for item in positive_focus),
            "canonical_gold_delete_focus_positions": sum(item.get("canonical_gold_label") == "$DELETE" for item in positive_focus),
            "canonical_delete_predicted_delete": sum(item.get("canonical_gold_label") == "$DELETE" and item["decoded_label"] == "$DELETE" for item in positive_focus),
            "canonical_keep_predicted_delete": sum(item.get("canonical_gold_label") == "$KEEP" and item["decoded_label"] == "$DELETE" for item in positive_focus),
            "positive_delete_in_raw_top_k": sum(item["delete_raw_rank"] is not None for item in positive_focus),
            "positive_mean_subword_pieces": sum(item["subword_piece_count"] for item in positive_focus) / len(positive_focus) if positive_focus else None,
            "positive_subword_piece_count_distribution": dict(Counter(str(item["subword_piece_count"]) for item in positive_focus)),
            "positive_mean_raw_delete_probability": sum(item["raw_delete_probability"] for item in positive_focus) / len(positive_focus) if positive_focus else None,
            "positive_mean_raw_keep_probability": sum(item["raw_keep_probability"] for item in positive_focus) / len(positive_focus) if positive_focus else None,
            "positive_delete_over_keep_bonus_margin": {
                "min": min(delete_keep_margins) if delete_keep_margins else None,
                "mean": sum(delete_keep_margins) / len(delete_keep_margins) if delete_keep_margins else None,
                "max": max(delete_keep_margins) if delete_keep_margins else None,
                "positive_count": sum(value > 0 for value in delete_keep_margins),
            },
            "positive_delete_rank_counts": dict(Counter(str(item["delete_raw_rank"] or "outside_top_k") for item in positive_focus)),
        },
        "segmentation_diagnostics": {
            "initial_sentence_slots_across_cases": sum(
                row.get("initial_sentence_slot_count", len(segment_sentences(row["source_text"]).sentences))
                for row in predictions
            ),
            "by_pass": segmentation_by_pass,
            "interpretation": "The production loop reuses initial sentence slots; fresh segmentation counts are observational outputs only.",
        },
        "authoritative_scorer": scorer_provenance(),
        "summary": result_summary,
        "limitations": [
            "The suite is a deterministic constructed stress sample, not an estimate of natural duplicate prevalence or production accuracy.",
            "Content/function classes use a declared hand-maintained function-word list; all selected rows remain unfiltered after inference.",
            "Exact text equality measures the generated intended repair; repeated identical surface tokens can make classifier-locus alignment inherently ambiguous.",
        ],
    }
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts["predictions_sha256"] = sha256_file(predictions_path)
    facts_path.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return facts


def _write_report(facts_path: Path, predictions_path: Path, report_path: Path) -> None:
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# FILLY duplicate-word inference stress diagnostic",
        "",
        f"Status: **{facts['status']}**. Suite size: **{facts['case_count']}** ({facts['summary']['case_counts'].get('new_duplicate_corruption', 0)} new corruptions, {sum(v for k, v in facts['summary']['case_counts'].items() if k.startswith('identity_'))} identity controls).",
        "",
        "This is a deterministic engineering stress test built from clean Stage-3 DEV target text. It does not estimate natural error prevalence or claim final-test performance.",
        "",
        "The existing saved synthetic 75-row duplicate benchmark is a separate historical result: its decoded `$DELETE` labels were TP=67, FP=2, FN=8 (P=0.971, R=0.893, F0.5=0.954), with 69/75 exact final texts. It was not rerun here; see [the evaluator findings](duplicate_evaluator_findings.md).",
        "",
        f"DEV SHA-256: `{facts.get('source', {}).get('sha256', 'see stress manifest')}`. Checkpoint SHA-256: `{facts['checkpoint']['sha256']}`. Encoder/tokenizer revision: `{facts['checkpoint']['model_revision']}`. Label vocabulary: {facts['checkpoint']['label_vocab_size']} labels, `{facts['checkpoint']['label_vocab_sha256']}`.",
        "",
        f"Current configured iteration count: **{facts['active_configured_iterations']}**. The current one-pass endpoint is P{facts['active_configured_output_pass']} from the same forward trace; P1–P5 were explicitly recorded. Model eval and inference mode verified: **{facts['runtime']['model_eval_mode_verified']} / {facts['runtime']['torch_inference_mode_verified']}**. Runtime `{facts['timing']['inference_seconds']:.2f}s` over {facts['runtime']['forward_calls']} model forwards.",
        "",
        "## Pass outcomes across all 120 rows",
        "",
        "The exact-match columns split the 100 constructed corruptions by token class and show the 20 identity controls separately. P/R/F0.5 and ERR in this table aggregate all 120 rows, including identity controls.",
        "",
        "| Pass | New corruptions exact | Function exact | Content exact | Identity repeats exact | Plain identities exact | P | R | F0.5 | ERR |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in facts["summary"]["by_pass"]:
        groups = item["groups"]
        classes = item.get("new_corruption_exact_by_class", {})
        def val(key: str, field: str = "exact_matches") -> str:
            value = groups.get(key, {}).get(field)
            return "—" if value is None else str(value)
        overall = item["overall_edit_metrics"]
        lines.append("| {pass_num} | {corrupt} | {function} | {content} | {repeat} | {plain} | {precision:.3f} | {recall:.3f} | {f05:.3f} | {err} |".format(
            pass_num=item["pass"],
            corrupt=val("new_duplicate_corruption"),
            function=(f"{classes.get('function', {}).get('exact_matches', 0)}/{classes.get('function', {}).get('n', 0)}"),
            content=(f"{classes.get('content', {}).get('exact_matches', 0)}/{classes.get('content', {}).get('n', 0)}"),
            repeat=val("identity_repeat"), plain=val("identity_plain"),
            precision=overall["precision"], recall=overall["recall"], f05=overall["f0_5"],
            err="—" if overall["err"] is None else f"{overall['err']:.3f}",
        ))
    lines.extend([
        "",
        "## Constructed-corruption edit metrics (100 rows only)",
        "",
        "These P/R/F0.5/ERR values exclude the 20 identity controls. P/R/F0.5 use the authoritative source-to-target versus source-to-output edit multiset scorer; ERR is 1 - residual target-relative edits / gold source-relative edits.",
        "",
        "| Pass | TP | FP | FN | Gold edits | Residual target edits | P | R | F0.5 | ERR | Exact target |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in facts["summary"]["by_pass"]:
        positive = item["groups"]["new_duplicate_corruption"]
        lines.append("| {pass_num} | {tp} | {fp} | {fn} | {gold} | {residual} | {p:.3f} | {r:.3f} | {f05:.3f} | {err:.3f} | {exact}/100 |".format(
            pass_num=item["pass"], tp=positive["tp"], fp=positive["fp"], fn=positive["fn"],
            gold=positive["gold_edits"], residual=positive["residual_edits"],
            p=positive["precision"], r=positive["recall"], f05=positive["f0_5"],
            err=positive["err"], exact=positive["exact_matches"],
        ))
    lines.extend([
        "",
        "The aggregate P1–P5 case counts and targeted token labels are in the machine-readable facts/predictions. P1 is the active one-pass service behavior; the five-pass run is diagnostic only. Each pass reuses sentence slots segmented from the original input, while the report records how a fresh segmentation would split each output.",
        "",
        "## First-subword top-k and decoder gates",
        "",
        f"Across both copies at each of {facts['topk_diagnostics']['positive_duplicate_focus_positions'] // 2} generated duplicate sites, raw correction argmax selected `$DELETE` for {facts['topk_diagnostics']['positive_raw_argmax_delete']}/{facts['topk_diagnostics']['positive_duplicate_focus_positions']}; `$DELETE` appeared in raw top-{facts['runtime']['top_k']} for {facts['topk_diagnostics']['positive_delete_in_raw_top_k']}; KEEP-adjusted argmax selected `$DELETE` for {facts['topk_diagnostics']['positive_keep_adjusted_argmax_delete']}; the KEEP bonus suppressed a raw `$DELETE` argmax {facts['topk_diagnostics']['positive_keep_bonus_suppressed_raw_delete']} times; the detection gate blocked it {facts['topk_diagnostics']['positive_detection_gate_blocked_delete']} times; the decoded label was `$DELETE` {facts['topk_diagnostics']['positive_decoded_delete']} times. Canonical alignment assigns `$DELETE` to {facts['topk_diagnostics']['canonical_gold_delete_focus_positions']} of these focus positions; decoded-delete matches there: {facts['topk_diagnostics']['canonical_delete_predicted_delete']}, versus extra `$DELETE` predictions on canonical KEEP positions: {facts['topk_diagnostics']['canonical_keep_predicted_delete']}. Mean source-token subword pieces: {facts['topk_diagnostics']['positive_mean_subword_pieces']} (distribution {facts['topk_diagnostics']['positive_subword_piece_count_distribution']}); mean raw P(DELETE)/P(KEEP): {facts['topk_diagnostics']['positive_mean_raw_delete_probability']:.4f}/{facts['topk_diagnostics']['positive_mean_raw_keep_probability']:.4f}.",
        "",
        f"The raw margin P(DELETE) - P(KEEP) - 0.2 had mean {facts['topk_diagnostics']['positive_delete_over_keep_bonus_margin']['mean']:.4f}, ranged {facts['topk_diagnostics']['positive_delete_over_keep_bonus_margin']['min']:.4f} to {facts['topk_diagnostics']['positive_delete_over_keep_bonus_margin']['max']:.4f}, and was positive at {facts['topk_diagnostics']['positive_delete_over_keep_bonus_margin']['positive_count']}/{facts['topk_diagnostics']['positive_duplicate_focus_positions']} focused positions. This margin is observational and does not capture competing correction labels.",
        "The predictor uses first-subword logits, softmax over correction labels, adds 0.2 to KEEP's score, then forces KEEP when detection error probability is below 0.5. There is no correction-label top-k decoder: top-k is read-only instrumentation from the same forward tensors.",
        "",
        "P/R/F0.5 use FILLY's authoritative final-output edit scorer on source, target, and each pass output. ERR is derived as 1 - residual target-relative edits / gold source-relative edits. Exact text matches and scorer counts are separate because either identical copy may be deleted to reach the same string. Each generated case's two adjacent copies carry canonical alignment labels. Identity-repeat controls are corpus rows with adjacent equal tokens; their original communicative intent was not manually verified. The suite does not include hand-written reduplication controls, which would fall outside the pinned DEV-source rule.",
        "",
        f"Fresh segmentation differed from the initial sentence slots in {sum(item['cases_with_fresh_segmentation_change'] for item in facts.get('segmentation_diagnostics', {}).get('by_pass', []))} case-pass outputs across P1 through P5. The production five-pass loop kept the initial slots; [duplicate_inference_facts.json](duplicate_inference_facts.json) records per-pass slot counts.",
        "",
        "## Provenance and files",
        "",
        "The sealed case-file SHA-256 is recorded in the manifest. Its `selection_rank` field predates the quota-rank audit and holds a candidate-level rank for positive cases; the sidecar records the actual group-and-length quota rank. The audit replay selected the same 100 case IDs and logged one canonical-replay rejection before replacement.",
        "",
        f"Checkpoint metadata: stage `{facts['checkpoint']['stage']}`, selected_best `{facts['checkpoint']['selected_best']}`, run `{facts['checkpoint']['run_id']}`, format {facts['checkpoint']['format_version']}; encoder `{facts['checkpoint']['model_id']}` revision `{facts['checkpoint']['model_revision']}`, Transformers `{facts['checkpoint']['transformers_version']}`. Training source-manifest SHA-256: `{facts['checkpoint']['source_manifest_hash']}`. Authoritative scorer SHA-256: `{facts['authoritative_scorer']['sha256']}`.",
        "",
        f"Checkpoint: `{facts['checkpoint']['path']}` ({facts['checkpoint']['size_bytes']} bytes). Predictions SHA-256: `{facts['predictions_sha256']}`. Case file: [duplicate_inference_stress_cases.jsonl](duplicate_inference_stress_cases.jsonl). Run facts: [duplicate_inference_facts.json](duplicate_inference_facts.json). Detailed traces: [duplicate_inference_predictions.jsonl](duplicate_inference_predictions.jsonl). Preparation manifest: [duplicate_inference_stress_manifest.json](duplicate_inference_stress_manifest.json). Rejected candidates: [duplicate_inference_rejections.json](duplicate_inference_rejections.json). Selection rank audit: [duplicate_inference_selection_ranks.json](duplicate_inference_selection_ranks.json).",
        "",
        "No final-test records were read or inferred. No production inference, normalization, or frontend code was changed.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="sample and generate new clean-DEV duplicate corruptions")
    prepare.add_argument("--source", type=Path, default=DEFAULT_DEV)
    prepare.add_argument("--cases", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_stress_cases.jsonl")
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_stress_manifest.json")
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    audit = subparsers.add_parser("audit", help="verify sealed case selection and write provenance sidecars without changing cases")
    audit.add_argument("--source", type=Path, default=DEFAULT_DEV)
    audit.add_argument("--cases", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_stress_cases.jsonl")
    audit.add_argument("--manifest", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_stress_manifest.json")
    run = subparsers.add_parser("run", help="run the selected FILLY checkpoint on prepared cases")
    run.add_argument("--cases", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_stress_cases.jsonl")
    run.add_argument("--manifest", type=Path, help="prepared case manifest; defaults to the case directory's stress manifest")
    run.add_argument("--predictions", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_predictions.jsonl")
    run.add_argument("--facts", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_facts.json")
    run.add_argument("--report", type=Path, default=DEFAULT_REPORTS / "duplicate_inference_report.md")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--limit-cases", type=int, help="run only the first N cases as a throughput benchmark; full runs require at least 100")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_cases(args.source, args.cases, args.manifest, seed=args.seed)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit":
        result = audit_case_provenance(args.source, args.cases, args.manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    facts = run_inference(args.cases, args.predictions, args.facts, top_k=args.top_k, limit_cases=args.limit_cases, manifest_path=args.manifest)
    _write_report(args.facts, args.predictions, args.report)
    print(json.dumps({"status": facts["status"], "case_count": facts["case_count"],
                      "inference_seconds": facts["timing"]["inference_seconds"],
                      "facts": str(args.facts), "report": str(args.report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
