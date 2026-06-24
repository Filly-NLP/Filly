import re
import logging
from typing import Optional

from app.schemas.schemas import NormalizationItem

logger = logging.getLogger(__name__)

# ─── Filipino Normalization Dictionary ──────────────────────────────
_SLANG_MAP: dict[str, str] = {
    # ── Pronouns ──
    "aq": "ako",
    "aq2": "ako",
    "ko": "ko",
    "u": "ikaw",
    "kau": "ikaw",
    "ikw": "ikaw",
    "nten": "niyan",
    "nyan": "niyan",
    "nyn": "niyan",
    "cla": "sila",
    "cya": "siya",
    "xa": "siya",
    "tyo": "tayo",
    "tau": "tayo",
    "kmi": "kami",
    "kme": "kami",

    # ── Common Words ──
    "nmn": "naman",
    "nmn2": "naman",
    "nman": "naman",
    "po": "po",
    "pu": "po",
    "pow": "po",
    "kc": "kasi",
    "kse": "kasi",
    "kasi2": "kasi",
    "kz": "kasi",
    "tlga": "talaga",
    "tlaga": "talaga",
    "tlg": "talaga",
    "cge": "sige",
    "sge": "sige",
    "cgi": "sige",
    "cguro": "siguro",
    "cgro": "siguro",
    "pra": "para",
    "pra2": "para",
    "pru": "para",
    "lng": "lang",
    "lng2": "lang",
    "lngg": "lang",
    "lam": "alam",
    "alm": "alam",
    "din": "din",
    "dn": "din",
    "den": "din",
    "dba": "di ba",
    "dbi": "di ba",
    "db": "di ba",
    "dpat": "dapat",
    "dpt": "dapat",
    "pde": "pwede",
    "pwd": "pwede",
    "pwde": "pwede",
    "pwdi": "pwede",
    "gsto": "gusto",
    "gsto2": "gusto",
    "gsto3": "gusto",
    "mganda": "maganda",
    "gnda": "maganda",

    # ── Greetings & Responses ──
    "musta": "kumusta",
    "kamusta": "kumusta",
    "oo": "oo",
    "opo": "opo",
    "opow": "opo",
    "tnx": "salamat",
    "ty": "salamat",
    "slmat": "salamat",
    "slmt": "salamat",
    "tnks": "salamat",
    "pki": "paki",

    # ── Verbs / Actions ──
    "gwa": "gawa",
    "gwa2": "gawa",
    "pnta": "punta",
    "pnta2": "punta",
    "pnta3": "punta",
    "pnts": "punta",
    "kain": "kain",
    "kn": "kain",
    "alis": "alis",
    "lis": "alis",
    "balik": "balik",
    "blik": "balik",

    # ── Adjectives / Adverbs ──
    "ang": "ang",
    "mhal": "mahal",
    "mhl": "mahal",
    "mgaling": "magaling",
    "mgling": "magaling",
    "nkakatawa": "nakakatawa",
    "nkktawa": "nakakatawa",
    "grbe": "grabe",
    "grabeh": "grabe",

    # ── Conjunctions / Particles ──
    "at": "at",
    "tpos": "tapos",
    "tps": "tapos",
    "pos": "tapos",
    "tas": "tapos",
    "pru": "pero",
    "pro": "pero",
    "pro2": "pero",
    "proo": "pero",
    "eh": "eh",
    "na": "na",
    "pa": "pa",
    "ba": "ba",

    # ── Question Words ──
    "ano": "ano",
    "anu": "ano",
    "anu2": "ano",
    "anong": "anong",
    "asan": "nasaan",
    "nasan": "nasaan",
    "san": "nasaan",
    "bkt": "bakit",
    "bkit": "bakit",
    "pano": "paano",
    "panu": "paano",
    "pano2": "paano",
    "panu2": "paano",
    "klan": "kailan",
    "kelan": "kailan",
    "kilan": "kailan",

    # ── Numbers / Time ──
    "isa": "isa",
    "is2": "isa",
    "dalawa": "dalawa",
    "dlwa": "dalawa",
    "tatlo": "tatlo",
    "ttlo": "tatlo",
    "mamaya": "mamaya",
    "mmya": "mamaya",
    "mya": "mamaya",
    "bkas": "bukas",
    "bkaz": "bukas",
    "kahapon": "kahapon",
    "khpon": "kahapon",
    "ngayon": "ngayon",
    "ngyn": "ngayon",
    "ngyn2": "ngayon",

    # ── Common Texting Shortcuts ──
    "hnd": "hindi",
    "hinde": "hindi",
    "hdi": "hindi",
    "di": "hindi",
    "wla": "wala",
    "wl": "wala",
    "meron": "meron",
    "mron": "meron",
    "mrn": "meron",
    "ung": "iyong",
    "yung": "iyong",
    "yng": "iyong",
    "ng": "ng",
}

_SKIP_WORDS: set[str] = {
    # Single-letter / very short function words
    "i", "a", "o", "e", "ni", "si", "ay", "sa", "ng", "na", "pa",
    "po", "ba", "at", "an", "ko", "ka", "mo", "to", "ta",
    # Common valid words
    "may", "ang", "mga", "din", "oo", "eh", "ano", "kain",
    "alis", "balik", "nang", "nag", "lang", "rin", "raw", "daw",
    "sila", "siya", "tayo", "kami", "ikaw", "ako", "naman",
    "kasi", "talaga", "sige", "siguro", "para", "alam",
    "gusto", "gawa", "punta", "mahal", "maganda", "magaling",
    "grabe", "tapos", "pero", "nasaan", "bakit", "paano",
    "kailan", "isa", "dalawa", "tatlo", "mamaya", "bukas",
    "kahapon", "ngayon", "hindi", "wala", "meron", "dapat",
    "pwede", "kumusta", "salamat", "opo", "paki",
}

_MIN_WORD_LENGTH = 2


def _damerau_levenshtein_distance(s1: str, s2: str) -> int:
    len1 = len(s1)
    len2 = len(s2)

    d: dict[tuple[int, int], int] = {}
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,       # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)

    return d[(len1 - 1, len2 - 1)]


def classify_normalization(word: str) -> str:
    word = word.lower()
    slangs = {
        "aq", "aq2", "u", "kau", "ikw", "nten", "nyan", "nyn", "cla", "cya", "xa", "tyo", "tau", "kmi", "kme",
        "po", "pu", "pow", "kc", "kse", "kz"
    }
    spelling_variations = {
        "nman", "kasi2", "sge", "cgi", "cguro", "cgro", "pra2", "pru", "lngg", "dba", "dbi", "db", "pwde",
        "pwdi", "gsto2", "gsto3", "mganda", "gnda", "kamusta", "opow", "gwa2", "pnta2", "pnta3", "pnts",
        "kn", "lis", "blik", "mhl", "mgling", "nkktawa", "grabeh", "tpos", "tps", "pos", "tas", "pro2",
        "proo", "anu", "anu2", "anong", "asan", "nasan", "san", "pano2", "panu", "panu2", "klan", "kelan",
        "kilan", "is2", "dlwa", "ttlo", "mmya", "mya", "bkas", "bkaz", "khpon", "ngyn", "ngyn2", "hinde",
        "hdi", "wl", "mrn", "ung", "yng"
    }
    if word in slangs:
        return "slang"
    if word in spelling_variations:
        return "spelling_variation"
    return "abbreviation"


class FilNormalizer:
    def __init__(
        self,
        custom_mappings: Optional[dict[str, str]] = None,
        max_edit_distance: int = 1,
        min_confidence: float = 0.6,
    ):
        self.mappings: dict[str, str] = {**_SLANG_MAP}
        if custom_mappings:
            self.mappings.update(custom_mappings)

        self.max_edit_distance = max_edit_distance
        self.min_confidence = min_confidence

        self._by_first_char: dict[str, list[str]] = {}
        for key in self.mappings:
            ch = key[0] if key else ""
            self._by_first_char.setdefault(ch, []).append(key)

    def normalize(self, text: str) -> list[NormalizationItem]:
        items: list[NormalizationItem] = []
        if not text or not text.strip():
            return items

        for match in re.finditer(r'\S+', text):
            raw_word = match.group()
            start = match.start()
            end = match.end()

            clean = re.sub(r'^[^\w]+|[^\w]+$', '', raw_word, flags=re.UNICODE)
            if not clean or len(clean) < _MIN_WORD_LENGTH:
                continue

            lower = clean.lower()

            if lower in _SKIP_WORDS:
                continue

            if lower in self.mappings:
                formal = self.mappings[lower]
                if lower != formal.lower():
                    offset = raw_word.lower().find(lower)
                    adj_start = start + offset
                    adj_end = adj_start + len(clean)
                    items.append(NormalizationItem(
                        word=clean,
                        suggestion=formal,
                        start=adj_start,
                        end=adj_end,
                        type="normalization",
                        confidence=1.0,
                        category=classify_normalization(lower),
                    ))
                continue

            if len(lower) <= 8:
                best_match = self._fuzzy_match(lower)
                if best_match:
                    formal, confidence = best_match
                    if confidence >= self.min_confidence and lower != formal.lower():
                        offset = raw_word.lower().find(lower)
                        adj_start = start + offset
                        adj_end = adj_start + len(clean)
                        items.append(NormalizationItem(
                            word=clean,
                            suggestion=formal,
                            start=adj_start,
                            end=adj_end,
                            type="normalization",
                            confidence=round(confidence, 2),
                            category=classify_normalization(lower),
                        ))

        return items

    def _fuzzy_match(self, word: str) -> Optional[tuple[str, float]]:
        best_key: Optional[str] = None
        best_dist = self.max_edit_distance + 1

        first_char = word[0] if word else ""
        candidates: list[str] = []
        for ch in {first_char, chr(ord(first_char) - 1) if first_char > 'a' else first_char,
                    chr(ord(first_char) + 1) if first_char < 'z' else first_char}:
            candidates.extend(self._by_first_char.get(ch, []))

        for key in self.mappings:
            if abs(len(key) - len(word)) <= self.max_edit_distance and key not in candidates:
                candidates.append(key)

        for key in candidates:
            if abs(len(key) - len(word)) > self.max_edit_distance:
                continue

            dist = _damerau_levenshtein_distance(word, key)
            if dist < best_dist:
                best_dist = dist
                best_key = key

        if best_key is not None and best_dist <= self.max_edit_distance:
            formal = self.mappings[best_key]
            confidence = max(0.0, 1.0 - (best_dist * 0.3))
            return formal, confidence

        return None


# ─── Module-level singleton ─────────────────────────────────────────
_normalizer: Optional[FilNormalizer] = None


def get_normalizer() -> FilNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = FilNormalizer()
        logger.info("Filipino normalizer initialized with %d mappings", len(_normalizer.mappings))
    return _normalizer
