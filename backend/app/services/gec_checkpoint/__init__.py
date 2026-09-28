"""Inference pieces mirrored from the checkpoint's training implementation.

The model, predictor, tokenizer, edit replay, and label registry here are kept
in sync with Grammar-Error-Correction commit
``a137350d1427ba473327a26c45b1425b8b553469``.  They are not the legacy
``backend/app/services/gector`` implementation, which expects another
checkpoint format and vocabulary.
"""
