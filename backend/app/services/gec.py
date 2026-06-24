import os
import re
import logging
from typing import Optional

from app.schemas.schemas import GrammarCorrectionItem

logger = logging.getLogger(__name__)


class GECService:
    def __init__(self):
        self.gector_model = None
        self.gector_tokenizer = None
        self.encode_verb_dict = {}
        self.decode_verb_dict = {}

        # Paths
        model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "gec", "models")
        )
        vocab_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "gec", "data", "output_vocabulary")
        )
        verb_vocab_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "gec", "data", "verb-form-vocab.txt")
        )

        model_path = None
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".pt") or f.endswith(".bin"):
                    model_path = os.path.join(model_dir, f)
                    break

        if model_path and os.path.exists(vocab_path):
            try:
                from app.services.gector.modeling import GECToR
                from app.services.gector.predict import load_verb_dict
                from transformers import AutoTokenizer

                logger.info(f"Initializing GECToR Tagalog model using: {model_path}")
                self.gector_tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-large")
                self.gector_model = GECToR.from_official_pretrained(
                    pretrained_model_name_or_path=model_path,
                    special_tokens_fix=1,
                    transformer_model="jcblaise/roberta-tagalog-large",
                    vocab_path=vocab_path
                )
                self.gector_model.eval()

                if os.path.exists(verb_vocab_file):
                    self.encode_verb_dict, self.decode_verb_dict = load_verb_dict(verb_vocab_file)
                logger.info("GECToR model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"GECToR model could not be loaded ({e}). Falling back to rule-based stub."
                )
                self.gector_model = None
                self.gector_tokenizer = None
        else:
            logger.info("No GECToR weights found. Using rule-based grammar checker.")

    def check(self, text: str) -> list[GrammarCorrectionItem]:
        if not text or not text.strip():
            return []

        if self.gector_model and self.gector_tokenizer:
            try:
                return self._check_gector(text)
            except Exception as e:
                logger.error(f"GECToR inference failed: {e}. Falling back to rule-based GEC.")

        corrections: list[GrammarCorrectionItem] = []

        # Run each rule checker
        corrections.extend(self._check_repeated_words(text))
        corrections.extend(self._check_din_rin_rule(text))
        corrections.extend(self._check_daw_raw_rule(text))
        corrections.extend(self._check_ng_nang(text))

        # Sort by position and remove overlapping corrections
        corrections.sort(key=lambda c: c.start)
        corrections = self._remove_overlaps(corrections)

        return corrections

    def _check_gector(self, text: str) -> list[GrammarCorrectionItem]:
        import torch
        from app.services.gector.predict import _predict, process_token

        words = list(re.finditer(r'\S+', text))
        if not words:
            return []

        tokens = [w.group() for w in words]
        srcs = [['$START'] + tokens]

        pred_labels, no_corrections = _predict(
            self.gector_model,
            self.gector_tokenizer,
            srcs,
            keep_confidence=0.0,
            min_error_prob=0.0,
            batch_size=1
        )

        if no_corrections[0]:
            return []

        corrections: list[GrammarCorrectionItem] = []
        labels = pred_labels[0]

        for i in range(1, len(labels)):
            if i - 1 >= len(tokens):
                break

            label = labels[i]
            if label in ['$KEEP', '<PAD>', '<OOV>']:
                continue

            word_match = words[i - 1]
            token = word_match.group()
            start = word_match.start()
            end = word_match.end()

            corrected = process_token(token, label, self.encode_verb_dict, self.decode_verb_dict)
            if corrected == token:
                continue

            if label == "$DELETE":
                corrections.append(GrammarCorrectionItem(
                    original=token,
                    correction="",
                    start=start,
                    end=end,
                    type="grammar",
                    rule="gector_delete",
                    message=f"Burahin ang salitang: '{token}'"
                ))
            elif "$APPEND_" in label:
                app = label.replace("$APPEND_", "")
                corrections.append(GrammarCorrectionItem(
                    original="",
                    correction=" " + app,
                    start=end,
                    end=end,
                    type="grammar",
                    rule="gector_append",
                    message=f"Idagdag ang '{app}' pagkatapos ng '{token}'"
                ))
            else:
                corrections.append(GrammarCorrectionItem(
                    original=token,
                    correction=corrected,
                    start=start,
                    end=end,
                    type="grammar",
                    rule="gector_replace",
                    message=f"Palitan ang '{token}' ng '{corrected}'"
                ))

        return corrections

    def _check_repeated_words(self, text: str) -> list[GrammarCorrectionItem]:
        items: list[GrammarCorrectionItem] = []
        pattern = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)

        for match in pattern.finditer(text):
            word = match.group(1)
            if len(word) < 2:
                continue

            items.append(GrammarCorrectionItem(
                original=match.group(),
                correction=word,
                start=match.start(),
                end=match.end(),
                type="grammar",
                rule="repeated_word",
                message=f'Salitang paulit-ulit: "{match.group()}" → "{word}"',
            ))

        return items

    def _check_din_rin_rule(self, text: str) -> list[GrammarCorrectionItem]:
        items: list[GrammarCorrectionItem] = []
        pattern = re.compile(
            r'([aeiouAEIOU])\s+(din)\b',
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            din_word = match.group(2)
            replacement = "Rin" if din_word[0].isupper() else "rin"
            din_start = match.start(2)
            din_end = match.end(2)

            items.append(GrammarCorrectionItem(
                original=din_word,
                correction=replacement,
                start=din_start,
                end=din_end,
                type="grammar",
                rule="din_rin",
                message='Pagkatapos ng patinig, gamitin ang "rin" sa halip na "din".',
            ))

        return items

    def _check_daw_raw_rule(self, text: str) -> list[GrammarCorrectionItem]:
        items: list[GrammarCorrectionItem] = []
        pattern = re.compile(
            r'([aeiouAEIOU])\s+(daw)\b',
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            daw_word = match.group(2)
            replacement = "Raw" if daw_word[0].isupper() else "raw"
            daw_start = match.start(2)
            daw_end = match.end(2)

            items.append(GrammarCorrectionItem(
                original=daw_word,
                correction=replacement,
                start=daw_start,
                end=daw_end,
                type="grammar",
                rule="daw_raw",
                message='Pagkatapos ng patinig, gamitin ang "raw" sa halip na "daw".',
            ))

        return items

    def _check_ng_nang(self, text: str) -> list[GrammarCorrectionItem]:
        items: list[GrammarCorrectionItem] = []
        noun_indicators = [
            "bata", "tao", "bahay", "aklat", "pera", "pagkain",
            "tubig", "mesa", "silya", "kotse", "telepono",
        ]

        for noun in noun_indicators:
            pattern = re.compile(
                rf'\bnang\s+({re.escape(noun)})\b',
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                items.append(GrammarCorrectionItem(
                    original=f"nang {match.group(1)}",
                    correction=f"ng {match.group(1)}",
                    start=match.start(),
                    end=match.end(),
                    type="grammar",
                    rule="ng_nang",
                    message='Gamitin ang "ng" bilang case marker bago ang pangngalan, hindi "nang".',
                ))

        return items

    @staticmethod
    def _remove_overlaps(
        corrections: list[GrammarCorrectionItem],
    ) -> list[GrammarCorrectionItem]:
        if not corrections:
            return corrections

        result: list[GrammarCorrectionItem] = [corrections[0]]
        for c in corrections[1:]:
            last = result[-1]
            if c.start >= last.end:
                result.append(c)
        return result


# ─── Module-level singleton ─────────────────────────────────────────
_gec_service: Optional[GECService] = None


def get_gec_service() -> GECService:
    global _gec_service
    if _gec_service is None:
        _gec_service = GECService()
        logger.info("GEC service initialized")
    return _gec_service
