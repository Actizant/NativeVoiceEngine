"""Training-side components. Deliberately separated from the product path.

Status while the architecture is under review: this package holds an isolated, dense,
unbanded **correctness reference** for the token-and-duration transducer objective. It is
not wired into any caller, no training loop imports it, and no checkpoint format exists, so
``GATED_TRAINING_CALLERS`` in ``tests/test_corpus_reader_fence.py`` remains empty.

Nothing here runs a model, reads a dataset, opens a network socket, or writes a file.
The frozen product contracts (``audio/features.py``, ``stt/api.py``) are not imported from
this package's numerics and are not changed by it.
"""
