"""Orchestrate FILLY's normalization then GEC inference stages."""

from __future__ import annotations

from typing import Any

from app.services.alignment import compose_suggestions
from app.schemas.schemas import (
    GECChangeResponse,
    GECCorrectResponse,
    GECIterationResponse,
    GECStageResponse,
    NormalizeResponse,
    PipelineAnalyzeResponse,
)


class FillyPipeline:
    """The single combined path from raw Filipino text to corrected text.

    Normalization changes have offsets into the original input. GEC changes
    retain their iteration-local offsets in ``gec.passes``; they are not
    presented as editor suggestions until alignment maps them to the original
    text.
    """

    def __init__(self, normalizer: Any, gec_service: Any, *, iterations: int = 5):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.normalizer = normalizer
        self.gec_service = gec_service
        self.iterations = iterations

    def normalize(self, text: str) -> NormalizeResponse:
        """Run only the normalization stage for research ablations."""
        if not isinstance(text, str):
            raise TypeError("FILLY input must be a string")
        normalized_text, changes = self.normalizer.normalize_text(text)
        return NormalizeResponse(
            original_text=text,
            normalized_text=normalized_text,
            changes=changes,
        )

    def _run_gec(self, text: str) -> GECStageResponse:
        result = self.gec_service.correct_iteratively(text, iterations=self.iterations)
        changes = [self._gec_change(change) for change in result.changes]
        passes = [
            GECIterationResponse(
                iteration=iteration.iteration,
                input_text=iteration.input_text,
                output_text=iteration.output_text,
                model_invoked=iteration.model_invoked,
                input_tokens=list(iteration.input_tokens),
                output_tokens=list(iteration.output_tokens),
                labels=list(iteration.labels),
                changes=[self._gec_change(change) for change in iteration.changes],
            )
            for iteration in result.passes
        ]
        return GECStageResponse(
            iterations=result.iterations,
            iteration_outputs=list(result.iteration_outputs),
            changes=changes,
            passes=passes,
            metadata=dict(result.metadata),
        )

    @staticmethod
    def _gec_change(change: Any) -> GECChangeResponse:
        return GECChangeResponse(
            original=change.original,
            replacement=change.correction,
            start=change.start,
            end=change.end,
            tag=change.label,
            confidence=change.confidence,
            iteration=change.iteration,
        )

    def gec(self, text: str) -> GECCorrectResponse:
        """Run only GEC inference, without normalizing its input."""
        if not isinstance(text, str):
            raise TypeError("FILLY input must be a string")
        stage = self._run_gec(text)
        return GECCorrectResponse(
            original_text=text,
            corrected_text=stage.passes[-1].output_text if stage.passes else text,
            gec=stage,
        )

    def analyze(self, text: str) -> PipelineAnalyzeResponse:
        """Run the production pipeline: original -> normalized -> GEC x N."""
        normalization = self.normalize(text)

        # This assignment is the ordering boundary: GEC receives exactly the
        # string returned by the normalizer, never the original input.
        gec = self._run_gec(normalization.normalized_text)
        corrected_text = gec.passes[-1].output_text if gec.passes else normalization.normalized_text

        suggestions = compose_suggestions(
            original_text=text,
            normalized_text=normalization.normalized_text,
            corrected_text=corrected_text,
            normalization_changes=normalization.changes,
            gec_passes=gec.passes,
        )

        return PipelineAnalyzeResponse(
            original_text=text,
            normalized_text=normalization.normalized_text,
            corrected_text=corrected_text,
            suggestions=suggestions,
            normalization=normalization,
            gec=gec,
        )
