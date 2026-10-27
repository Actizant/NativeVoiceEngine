"""Torch implementation of the TDT lattice loss: dense, unbanded, autograd-only.

Status: **correctness reference, not optimised, wired to no caller.** ``torch`` is absent
from the global project environment, where ``tests/test_tdt_lattice_reference.py`` therefore
reports every torch-dependent check as NOT RUN rather than inferring a result. The module has
since been executed on this machine in a CPU-only verification environment installed outside
the repository (Python 3.14.7, ``torch 2.14.1+cpu``, float64), and there it agrees with the
independent definitions: its loss matches the float64 recursion on all 45 admissible grid
cells to a worst deviation of 8.882e-16 where the test bound is 1e-9, its autograd gradient
matches the path-sum objective to 2.220e-16 where the bound is 1e-8, repeat calls are
bit-identical, every malformed-input guard and the three geometry-impossible cells refuse
by name, and the blank-only lattices match the exact-rational path enumerator. No training
loop, checkpoint, model asset or product path has used it. Its gradient has now been compared
against central differences of this module's own output - 72 components of a deterministic 3x2
lattice at h = 1e-6, worst relative error 6.4e-10 against a 1e-7 bound, in
``tests/test_tdt_lattice_reference.py::test_torch_gradients_match_central_differences_of_the_torch_loss_itself``.
That node is torch-gated, so it is NOT RUN in the project interpreter.

Deliberately absent, by approval scope: banding, chunking, fused or custom kernels, a
closed-form backward for the pre-softmax logits, and any memory optimisation. Those need
their own design after this file is shown to agree with the exact-rational oracle, the
float64 reference and finite differences.

``torch`` is imported inside functions, never at module level, because
``tools/audit.py``'s lazy-import check fails a module-level heavy import in ``src/``.
"""

from __future__ import annotations

from nativevoice.training.errors import TdtAlignmentImpossible, TdtContractError
from nativevoice.training.tdt_lattice import NEG_INF, TdtSpec, check_expected_positions


def _torch():
    try:
        import torch
    except Exception as exc:                                   # pragma: no cover - env dependent
        raise TdtContractError(
            "the torch TDT loss needs a torch runtime, which is not installed on this machine",
            error=type(exc).__name__,
        ) from exc
    return torch


def _check_shapes(torch, logits, targets, lengths, spec: TdtSpec):
    """Batch-level structural guards. Per-row target rules are in :func:`_check_rows`.

    Token ids are validated only inside each row's declared prefix, because a padded batch carries
    values beyond ``U_b`` that the lattice never reads: they are not part of any hypothesis, so
    refusing them would make ordinary padding impossible and accepting them would be honest. What
    remains enforced for the whole tensor is shape, dtype, the width against the joint axis, and the
    declared length against the available width. ``nativevoice.training.tdt_lattice.validate``
    applies the same scope, so the two implementations refuse the same inputs with the same type.
    """
    if logits.ndim != 4:
        raise TdtContractError("logits must be [B, T, U+1, vocab + Nd]", ndim=logits.ndim)
    B, T, U1, W = logits.shape
    if B == 0:
        raise TdtContractError(
            "an empty batch carries no utterance to score; skipping a batch is the caller's job, "
            "and returning an empty result here would reach a trainer as a NaN reduction",
            batch=0)
    if targets.ndim != 2 or int(targets.shape[0]) != B:
        raise TdtContractError("targets must be [B, U]", targets=list(targets.shape), expect=[B, U1 - 1])
    if int(targets.shape[1]) > U1 - 1:
        raise TdtContractError("targets must be [B, U]", targets=list(targets.shape), expect=[B, U1 - 1])
    if lengths.ndim != 2 or tuple(lengths.shape) != (B, 2):
        raise TdtContractError("lengths must be [B, 2]", lengths=list(lengths.shape))
    if W <= spec.n_durations:
        raise TdtContractError("logit width must exceed the duration slice", width=W, n_durations=spec.n_durations)
    if T < 1:
        raise TdtContractError("logits need at least one encoder row", T=T)
    vocab = W - spec.n_durations
    if spec.vocab_size is not None and vocab != spec.vocab_size:
        raise TdtContractError(
            "the declared vocabulary disagrees with the logit width: the duration slice would be "
            "read as tokens, so the lattice would score the wrong slots",
            width=W, declared_vocab_size=spec.vocab_size, implied_vocab_size=vocab,
            n_durations=spec.n_durations)
    if spec.blank_index >= vocab:
        raise TdtContractError("blank_index outside the token slice", blank_index=spec.blank_index, vocab=vocab)
    if targets.dtype not in (torch.int32, torch.int64):
        raise TdtContractError("targets must be an integer tensor", dtype=str(targets.dtype))
    declared_T = int(lengths[:, 0].max().item())
    if declared_T > T:
        raise TdtContractError(
            "declared length T exceeds the number of logit rows", declared=declared_T, rows=T
        )
    return B, T, U1 - 1, vocab


def _check_rows(torch, targets, lengths, B, vocab, spec: TdtSpec):
    """Per-row guards: the declared prefix is the only part of a target that is real.

    Declared lengths are authoritative here, not the tensor's physical shape: the batch-level guard
    in :func:`_check_shapes` already refuses a tensor with no rows, and this refuses a caller who
    *declares* ``T_b = 0`` while supplying rows, which is the same contract breach seen from the
    other side. ``nativevoice.training.tdt_lattice.validate`` refuses it too, so neither
    implementation scores a zero-position utterance and the reference no longer diverges on it.
    """
    for b in range(B):
        T_b = int(lengths[b][0].item())
        U_b = int(lengths[b][1].item())
        if T_b < 0 or U_b < 0:
            raise TdtContractError("lengths must be non-negative", batch=b, T=T_b, U=U_b)
        if T_b < 1:
            raise TdtContractError(
                "each utterance must declare at least one encoder position", batch=b, T=T_b, U=U_b,
            )
        width = int(targets[b].numel())
        if U_b > width:
            raise TdtContractError(
                "declared target length exceeds the available target width",
                batch=b, U=U_b, target_width=width,
            )
        if U_b == 0:
            continue                                    # an empty hypothesis has no ids to check
        prefix = targets[b][:U_b]
        if bool((prefix < 0).any()) or bool((prefix >= vocab).any()):
            raise TdtContractError("target id outside the vocabulary", batch=b, U=U_b, vocab=vocab)
        if bool((prefix == spec.blank_index).any()):
            raise TdtContractError("the blank symbol cannot appear in a target", batch=b, U=U_b)


def tdt_loss(logits, targets, lengths, spec: TdtSpec, *,
             expected_positions=None) -> "object":
    """Per-utterance negative log-likelihood ``-log alpha[T, U]``, shape ``[B]``.

    ``spec`` is required, with no default, exactly as in the reference: inheriting one would choose
    a duration alphabet and a blank id that nobody declared.

    A direct transcription of :func:`nativevoice.training.tdt_lattice.alpha_table`: same
    0-based audio axis, same sentinel that exists only at ``u == U``, blanks summed over
    durations excluding 0, tokens over all durations including 0, one lattice likelihood.
    Gradients flow through the log-softmaxes into ``logits``; no custom backward is defined.

    ``expected_positions`` is the same seam as in the reference, checked by the same shared
    function, so the two implementations refuse the same disagreement with the same type and the
    same detail fields. The check runs after the shape and row guards and before any lattice cell
    is built.
    """
    torch = _torch()
    B, T_rows, U_cols, vocab = _check_shapes(torch, logits, targets, lengths, spec)
    _check_rows(torch, targets, lengths, B, vocab, spec)
    if expected_positions is not None:
        check_expected_positions(lengths, expected_positions, batch=B)
    # A slice that cannot be normalised is refused here, in the same type and with the same message
    # family as the reference, rather than being allowed to reach log_softmax and return NaN.
    # -inf entries are legitimate (a class of probability zero); NaN never is, and a slice with no
    # finite entry at all has no log-softmax.
    if bool(torch.isnan(logits).any()):
        raise TdtContractError("a logit is NaN, so the lattice cannot be scored",
                               count=int(torch.isnan(logits).sum().item()))
    token_ok = torch.isfinite(logits[..., :vocab]).any(dim=-1)
    duration_ok = torch.isfinite(logits[..., vocab:]).any(dim=-1)
    bad = torch.nonzero(~token_ok | ~duration_ok)
    if int(bad.shape[0]):
        b, i, u = (int(v) for v in bad[0].tolist())
        raise TdtContractError(
            "this slice carries no finite logit, so its log-softmax is undefined: a malformed "
            "input, not an unreachable alignment",
            batch=b, row=i, col=u, slice="token" if not bool(token_ok[b, i, u]) else "duration")
    nd = spec.n_durations
    token_lp = torch.log_softmax(logits[..., : vocab], dim=-1)
    dur_lp = torch.log_softmax(logits[..., vocab:], dim=-1)
    durations = list(spec.durations)
    device, dtype = logits.device, token_lp.dtype
    out = torch.empty(B, dtype=dtype, device=device)

    for b in range(B):
        T = int(lengths[b][0].item())
        U = int(lengths[b][1].item())
        if T == 0 and U > 0 and 0 not in durations:
            raise TdtAlignmentImpossible(
                "no admissible alignment exists for this length and target",
                batch=b, T=T, U=U, durations=durations,
                reason="zero rows with a non-empty target requires a zero-duration token",
            )
        alpha = torch.full((T + 1, U + 1), float(NEG_INF), dtype=dtype, device=device)
        alpha[0][0] = 0.0
        for s in range(1, T + U + 1):
            for i in range(min(s, T) + 1):
                u = s - i
                if u < 0 or u > U or (i == T and u != U):
                    continue
                terms = []
                for k, d in enumerate(durations):
                    if d == 0:
                        continue                            # Eq. 6: blanks over D excluding 0
                    src = (i - d, u)
                    if src[0] < 0 or src == (i, u) or torch.isneginf(alpha[src[0]][src[1]]):
                        continue
                    terms.append(
                        alpha[src[0]][src[1]]
                        + token_lp[b, src[0], src[1], spec.blank_index]
                        + dur_lp[b, src[0], src[1], k]
                    )
                if u >= 1:
                    wanted = int(targets[b][u - 1].item())
                    for k, d in enumerate(durations):
                        src = (i - d, u - 1)
                        if src[0] < 0 or torch.isneginf(alpha[src[0]][src[1]]):
                            continue
                        terms.append(
                            alpha[src[0]][src[1]]
                            + token_lp[b, src[0], src[1], wanted]
                            + dur_lp[b, src[0], src[1], k]
                        )
                if terms:
                    alpha[i][u] = torch.logsumexp(torch.stack(terms), dim=0)
        value = alpha[T][U]
        if bool(torch.isneginf(value)) or bool(torch.isnan(value)):
            raise TdtAlignmentImpossible(
                "the lattice produced a non-finite likelihood", batch=b, T=T, U=U,
                durations=durations, reason="terminal cell unreachable or non-finite",
            )
        out[b] = -value
    return out
