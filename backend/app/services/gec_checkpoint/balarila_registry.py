"""Frozen FILLY/Balarila correction metadata copied from sibling GEG resources.

The source registry is read-only in ``Grammar-Error-Generator``.  This module is
the small, versioned runtime mirror used by FILLY GEC; it deliberately does not
import GEG code or claim that every tag has a candidate-generation resource.
The sibling capacity report marks hyphen, space, and morphology unavailable.
Those tags are usable here only with concrete source/target evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


REGISTRY_VERSION = "filly-balarila-table2-v1"
TABLE3_VERSION = "filly-balarila-table3-v1"
SOURCE_TAGS_SHA256 = "bdd62bd2bf71d781da50c5b4d500a05315823e9ee9c8a68196fa285259328cca"
SOURCE_TABLE3_SHA256 = "581cf03a3d1b99ff36a0351eb627132aface5628b7ee0b8f225571ca6808a98e"


@dataclass(frozen=True, slots=True)
class BalarilaTag:
    id: str
    family: str


# Preserves the ordered ids and family assignments in resources/balarila_tags.yaml.
_TAGS = (
    ("$REPLACE_nang", "ng_nang"), ("$REPLACE_ng", "ng_nang"),
    ("$REPLACE_daw", "enclitic"), ("$REPLACE_din", "enclitic"),
    ("$REPLACE_dito", "enclitic"), ("$REPLACE_diyan", "enclitic"),
    ("$REPLACE_doon", "enclitic"), ("$REPLACE_raw", "enclitic"),
    ("$REPLACE_rin", "enclitic"), ("$REPLACE_roon", "enclitic"),
    ("$REPLACE_rito", "enclitic"), ("$REPLACE_riyan", "enclitic"),
    ("$MERGE_HYPHEN", "hyphen"), ("$TRANSFORM_INSERT_HYPHEN", "hyphen"),
    ("$TRANSFORM_SPLIT_HYPHEN", "hyphen"), ("$MERGE_SPACE", "space"),
    ("$TRANSFORM_SPLIT_SPACE", "space"), ("$DELETE", "duplicate_word"),
    ("$TRANSFORM_VERB_BASE", "morphology"),
    ("$TRANSFORM_VERB_COMPACT", "morphology"),
    ("$TRANSFORM_VERB_COMPOBJ", "morphology"),
    ("$TRANSFORM_VERB_CONTACT", "morphology"),
    ("$TRANSFORM_VERB_CONTOBJ", "morphology"),
    ("$TRANSFORM_VERB_INCACT", "morphology"),
    ("$TRANSFORM_VERB_INCOBJ", "morphology"),
    ("$TRANSFORM_VERB_RECCOMP", "morphology"),
    ("$ADD_PUNC_EMARK", "punctuation"), ("$ADD_PUNC_PERIOD", "punctuation"),
    ("$ADD_PUNC_QMARK", "punctuation"), ("$CHANGE_PUNC_EMARK", "punctuation"),
    ("$CHANGE_PUNC_PERIOD", "punctuation"), ("$CHANGE_PUNC_QMARK", "punctuation"),
    ("$TRANSFORM_CASE_CAPITAL", "casing"), ("$TRANSFORM_CASE_LOWER", "casing"),
    ("$APPEND_t1", "missing_word"), ("$REPLACE_nila", "pronoun"),
    ("$REPLACE_niya", "pronoun"), ("$REPLACE_sila", "pronoun"),
    ("$REPLACE_siya", "pronoun"),
)

REGISTRY = tuple(BalarilaTag(*entry) for entry in _TAGS)
TAG_TO_FAMILY = {tag.id: tag.family for tag in REGISTRY}

if len(REGISTRY) != 39 or len(TAG_TO_FAMILY) != len(REGISTRY):
    raise RuntimeError("embedded Balarila Table 2 registry is incomplete or duplicated")

TABLE3_TARGET_TAGS = frozenset(
    {
        "$TRANSFORM_VERB_BASE", "$TRANSFORM_VERB_COMPACT", "$TRANSFORM_VERB_COMPOBJ",
        "$TRANSFORM_VERB_CONTACT", "$TRANSFORM_VERB_CONTOBJ", "$TRANSFORM_VERB_INCACT",
        "$TRANSFORM_VERB_INCOBJ", "$TRANSFORM_VERB_RECCOMP",
    }
)


def registered_tag(tag_id: str) -> BalarilaTag | None:
    return next((tag for tag in REGISTRY if tag.id == tag_id), None)


def tag_family(tag_id: str) -> str | None:
    return TAG_TO_FAMILY.get(tag_id)
