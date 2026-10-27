"""Dense, unbanded float64 reference for the token-and-duration transducer (TDT) objective.

This module is the correctness definition. It is deliberately independent of
``tdt_loss.py`` (which uses autograd) and of ``tests/tdt_lattice_oracle.py`` (which uses
exact rationals): three implementations of one function, none sharing its arithmetic, so a
shared bug cannot certify itself.

Indexing, and the one assumption it rests on
--------------------------------------------
The source writes the lattice 1-based: states ``(t, u)`` with ``t in [1, T+1]``, base
condition ``alpha(1,0) = 1``, backward base ``beta(T+1,U) = 1``, and
``beta(T+1,u) = alpha(T+1,u) = 0`` for every ``u != U``, with ``P_TDT(y|x) = alpha(T+1,U)``.

Here the audio axis is 0-based: ``i = t - 1``, so ``i in [0, T]``, the base cell is
``alpha[0][0]``, and the terminal node is ``(T, U)``. Row ``i = T`` is the terminal
**sentinel**: no logit is ever read there, because every transition into it comes from
``i - d`` with ``d >= 1``. A state ``(T, u)`` with ``u != U`` is therefore inadmissible.

ASSUMPTION A1, SPECIFIED AND PROVISIONALLY RULED for this implementation, still unresolved against
the paper. That ``T`` counts encoder output positions and the sentinel carries no acoustic state is
what those boundary sentences and the Appendix D joint shape ``T x U x (V+1+Nd)`` jointly imply; the
paper never states it, and six retrieval routes failed to produce a definition. The clauses recorded
on 2026-10-27 are S1 (``T`` and ``T_b`` count encoder output positions, and every duration is
measured in that same unit), S2 (the terminal node ``(T, U)`` is virtual: exactly ``T`` acoustic
rows, row ``T`` reads no logit, ``(T, u)`` inadmissible for ``u != U``), S3 (tokens over all ``D``
including 0, blanks over ``D`` excluding 0, ``sum D == T`` exactly, both closure families carrying
probability) and S5-S8 (``D``/``Nd`` ruled only after the subsampling stack; declared lengths may not
exceed physical extent while physical extent may inertly exceed declared length; ``expected_positions``
is optional, so enforcement is caller-conditional; and none of this is evidence about the authors'
intent). ``test_the_sentinel_row_is_never_read`` in ``tests/test_tdt_lattice_reference.py`` is the
observable consequence inside this module and ``tests/test_tdt_length_seam.py`` pins the seam; neither
can prove the convention is right - only that the implementation is consistent with it - so the length
checks in :func:`validate` and the comparison in :func:`check_expected_positions` are both required.
A frame-indexed ``T`` with every duration scaled by the stack's stride is provably isomorphic to this
reading, so the two cannot be told apart by any lattice test. If the recurrence can never satisfy the
oracle and this convention at the same time, that is a contradiction to report, not a convention to
change.

Semantics, verified from the source (Eq. 5/6/7/8/9)
---------------------------------------------------
* One log-softmax per slice: ``P(v, d | i, u) = P_T(v | i, u) * P_D(d | i, u)``.
* A token emission appends one target id and advances the audio axis by ``D[k]`` frames.
* A blank appends nothing and advances by ``D[k]`` frames.
* Blanks are summed over durations excluding 0; tokens over all durations, **including 0** -
  the zero-duration token is what makes a target longer than the frame count representable.
* The loss is one lattice likelihood. No separate duration term, no weight, no ``omega``:
  in the source ``omega`` is a loss-sampling probability, which belongs to a caller.

Keyword arguments named ``mutate_*``, plus ``reachable``, exist only so the test suite can
show each rule is load-bearing. No production caller passes them, and none of them is an
optimisation: banding, chunking and fused kernels are deliberately absent from this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from nativevoice.training.errors import TdtAlignmentImpossible, TdtContractError

NEG_INF = float("-inf")


@dataclass(frozen=True, slots=True)
class TdtSpec:
    """The shape of the lattice, independent of any tensor library.

    ``vocab_size`` is the declared width of the token slice. It is optional because the reference
    exists to score bare tensors, but a caller that knows its model's vocabulary MUST declare it:
    without it the vocabulary is inferred from ``width - len(durations)``, so a spec whose duration
    alphabet is shorter than the model's re-labels duration slots as token ids and still returns a
    finite loss. Declaring it turns that silent corruption into a refusal.
    """

    durations: tuple[int, ...] = (0, 1, 2)
    blank_index: int = 0
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        d = tuple(self.durations)
        object.__setattr__(self, "durations", d)
        if not d:
            raise TdtContractError("durations must not be empty")
        if any(not isinstance(x, int) or isinstance(x, bool) for x in d):
            raise TdtContractError("durations must be integers")
        if min(d) < 0:
            raise TdtContractError("durations must be non-negative", durations=list(d))
        if list(d) != sorted(set(d)):
            raise TdtContractError("durations must be strictly increasing and distinct", durations=list(d))
        if not any(x > 0 for x in d):
            raise TdtContractError(
                "durations must contain at least one positive value, or the lattice cannot advance",
                durations=list(d),
            )
        if self.blank_index < 0:
            raise TdtContractError("blank_index must be non-negative", blank_index=self.blank_index)
        if self.vocab_size is not None:
            if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int):
                raise TdtContractError("vocab_size must be an integer or None", vocab_size=self.vocab_size)
            if self.vocab_size < 1:
                raise TdtContractError("vocab_size must be at least 1", vocab_size=self.vocab_size)
            if self.blank_index >= self.vocab_size:
                raise TdtContractError(
                    "blank_index must sit inside the declared token slice",
                    blank_index=self.blank_index, vocab_size=self.vocab_size)

    @property
    def n_durations(self) -> int:
        return len(self.durations)

    @property
    def positive_durations(self) -> tuple[int, ...]:
        return tuple(x for x in self.durations if x > 0)

    @property
    def logit_width(self) -> int | None:
        """The emission width this spec implies, or ``None`` when the vocabulary is undeclared."""
        return None if self.vocab_size is None else self.vocab_size + self.n_durations


def logsumexp(values: Sequence[float]) -> float:
    """Stable sum in the log domain. Empty input is the additive identity, ``-inf``."""
    best = NEG_INF
    for v in values:
        if v > best:
            best = v
    if best == NEG_INF:
        return NEG_INF
    total = math.fsum(math.exp(v - best) for v in values if v != NEG_INF)
    if total <= 0.0:
        return NEG_INF
    return best + math.log(total)


def log_softmax(values: Sequence[float]) -> list[float]:
    shift = max(values)
    total = math.fsum(math.exp(v - shift) for v in values)
    offset = shift + math.log(total)
    return [v - offset for v in values]


def validate(logits, targets, lengths, spec: TdtSpec) -> int:
    """Check the caller's contract and return the batch size.

    ``logits[b][i][u]`` is a flat sequence of ``vocab + Nd`` logits: token slice first,
    duration slice last. Rows are indexed, never copied, so a lazy sequence is accepted.

    ``T == 0`` is refused outright, for every target length and every duration set. It is not
    a degenerate lattice: a zero-position encoder output means the caller has nothing to align
    to, and admitting it previously produced three different outcomes from one call site -
    ``-0.0`` for an empty target, ``TdtAlignmentImpossible`` for an unreachable one, and an
    uncaught ``IndexError`` whenever ``0 in D`` and ``U == 1`` (the sentinel rule makes every
    intermediate cell inadmissible at ``T == 0``, so ``(0, 1)`` was the only cell that could
    reach for a row that had never been required to exist). This now matches the torch
    transcription, which had always refused it.

    Token ids are validated only inside each row's declared prefix - ``range(U)`` below - and the
    torch transcription now scopes its check the same way. Values beyond ``U_b`` belong to no
    hypothesis, are never read by the lattice, and are therefore not token content: a padded batch
    may hold anything there, including ids that would be illegal as tokens. Two structural facts
    are still enforced: the declared length may not exceed the available target width, and the
    target width may not exceed the lattice's prediction axis. One asymmetry is inherent rather
    than chosen - a Python sequence carries no dtype, so the integer-tensor guard can exist only
    on the torch side.
    """
    if len(logits) != len(targets) or len(logits) != len(lengths):
        raise TdtContractError(
            "logits, targets and lengths must share a batch size",
            logits=len(logits), targets=len(targets), lengths=len(lengths),
        )
    if len(logits) == 0:
        raise TdtContractError(
            "an empty batch carries no utterance to score; skipping a batch is the caller's job, "
            "and returning an empty result here would reach a trainer as a NaN reduction",
            batch=0)
    vocab = None
    for b in range(len(logits)):
        T, U = int(lengths[b][0]), int(lengths[b][1])
        if T < 0 or U < 0:
            raise TdtContractError("lengths must be non-negative", batch=b, T=T, U=U)
        if len(targets[b]) < U:
            raise TdtContractError("targets shorter than the declared length", batch=b, U=U, given=len(targets[b]))
        if T == 0:
            raise TdtContractError(
                "the lattice needs at least one encoder position; T = 0 is a caller error, "
                "not an empty lattice", batch=b, T=T, U=U, rows=len(logits[b]),
            )
        if len(logits[b]) < T:
            raise TdtContractError(
                "fewer logit rows than the declared length T", batch=b, T=T, rows=len(logits[b])
            )
        for i in range(T):
            if len(logits[b][i]) < U + 1:
                raise TdtContractError(
                    "logit row too short for the declared target length",
                    batch=b, row=i, need=U + 1, given=len(logits[b][i]),
                )
            if len(targets[b]) > len(logits[b][i]) - 1:
                raise TdtContractError(
                    "targets wider than the lattice's prediction axis",
                    batch=b, row=i, targets=len(targets[b]), capacity=len(logits[b][i]) - 1,
                )
            width = len(logits[b][i][0])
            if width <= spec.n_durations:
                raise TdtContractError(
                    "logit width must exceed the duration slice",
                    batch=b, row=i, width=width, n_durations=spec.n_durations,
                )
            this_vocab = width - spec.n_durations
            if spec.vocab_size is not None and this_vocab != spec.vocab_size:
                raise TdtContractError(
                    "the declared vocabulary disagrees with the logit width: the duration slice "
                    "would be read as tokens, so the lattice would score the wrong slots",
                    batch=b, row=i, tensor_width=width, declared_vocab_size=spec.vocab_size,
                    implied_vocab_size=this_vocab, n_durations=spec.n_durations,
                )
            if vocab is None:
                vocab = this_vocab
            elif this_vocab != vocab:
                raise TdtContractError("logit width varies across the lattice", batch=b, row=i)
            for u in range(1, U + 1):
                if len(logits[b][i][u]) != width:
                    raise TdtContractError("ragged logit width within a row", batch=b, row=i, col=u)
        if vocab is not None:
            if spec.blank_index >= vocab:
                raise TdtContractError(
                    "blank_index outside the token slice", blank_index=spec.blank_index, vocab=vocab
                )
            for u in range(U):
                tid = int(targets[b][u])
                if tid < 0 or tid >= vocab:
                    raise TdtContractError("target id outside the vocabulary", batch=b, u=u, id=tid, vocab=vocab)
                if tid == spec.blank_index:
                    raise TdtContractError("the blank symbol cannot appear in a target", batch=b, u=u)
    return len(logits)


def check_expected_positions(lengths, expected_positions, *, batch: int) -> tuple[int, ...]:
    """Refuse any declared ``T_b`` that disagrees with the seam-derived position count.

    One checker, shared by both implementations through ``nativevoice.training.tdt_loss``, so the
    reference and the autograd transcription cannot drift in message, detail or exception type.
    ``lengths`` may be a sequence of sequences or a ``[B, 2]`` tensor; only ``lengths[b][0]`` is
    consulted, and never written.

    The authority is ``nativevoice.modeling.masking.position_lengths_from_audio`` - sample counts
    through the frozen frontend and the declared ``StackSpec`` into per-row position counts. There
    is no second producer and no alias: that route is the contract, and this function only compares
    a caller's declaration to it.

    Why a comparison is the only possible shape of this guard is worth stating plainly. Extra logit
    rows are inert by construction (the terminal sentinel reads nothing at index ``T``), so an
    under-declared ``T_b`` leaves no trace in the logits for any later check to find - measured:
    one utterance scored with ``T_b`` of 49, 47, 43 and 1 returns four different finite losses and
    refuses none. The declared length therefore has to be checked against something outside the
    tensor, and that is what ``expected_positions`` is for.

    Its limit is measured, not assumed: the comparison sees integers only, so it cannot tell which
    stack produced them. A two-stage model whose rows were derived from a one-stage stack was scored
    at ``T = 10`` instead of ``T_b = 5`` with the seam fully supplied, and the loss accepted it
    (10.957882927 against 5.774457903 for the same row). This guard therefore verifies a caller's
    arithmetic, not the caller's architecture, and it is NOT training-safe until a contract carries
    the frontend and the ``StackSpec`` themselves.

    Input contract: an iterable of Python ``int``s, and ``position_lengths_from_audio`` returns
    exactly that. Text is refused instead of iterated character by character; ``bool`` is refused
    even though it subclasses ``int``; floats and tensor scalars are refused. Nothing is coerced.
    ``lengths[b][0]`` *is* read through ``int()`` because a ``[B, 2]`` tensor is a legitimate
    ``lengths`` on the torch path, while ``expected_positions`` stays a Python-int sequence so this
    reference module needs no tensor library and a caller cannot smuggle a float or a 0-d tensor
    past the comparison. A generator or ``range`` is accepted - it is materialised once, here.
    """
    if isinstance(expected_positions, (str, bytes)):
        raise TdtContractError("expected_positions must be a sequence of integers, not text",
                               expected_positions=expected_positions)
    try:
        expected = tuple(expected_positions)
    except TypeError as exc:
        raise TdtContractError("expected_positions must be a sequence of integers",
                               expected_positions=str(expected_positions)) from exc
    if len(expected) != batch:
        raise TdtContractError("expected_positions must name every row of the batch exactly once",
                               rows=len(expected), batch=batch)
    for b, want in enumerate(expected):
        if isinstance(want, bool) or not isinstance(want, int):
            raise TdtContractError("expected_positions must be integer position counts",
                                   batch=b, expected_positions=want)
        if want < 1:
            raise TdtContractError(
                "the seam says this audio yields no encoder position; a zero-position row is not "
                "scorable", batch=b, expected_T=want)
        declared = int(lengths[b][0])
        if declared != want:
            raise TdtContractError(
                "declared encoder length disagrees with the frontend-derived position count",
                batch=b, declared_T=declared, expected_T=want)
    return expected


def _slices_for(logits_b, spec: TdtSpec, batch: int) -> Callable[[int, int], tuple[list[float], list[float]]]:
    """Log-softmax of each slice, computed once per ``(i, u)``.

    A slice is refused when it cannot be normalised, rather than being allowed to yield NaN that
    later surfaces as an "unreachable alignment". A ``-inf`` logit is legitimate - it means a class
    of probability zero - so the refusal is for NaN anywhere, and for a slice with no finite entry
    at all, which is the only case where the log-softmax is genuinely undefined.
    """
    cache: dict[tuple[int, int], tuple[list[float], list[float]]] = {}

    def at(i: int, u: int) -> tuple[list[float], list[float]]:
        key = (i, u)
        hit = cache.get(key)
        if hit is None:
            raw = logits_b[i][u]
            n_tok = len(raw) - spec.n_durations
            for label, part in (("token", raw[:n_tok]), ("duration", raw[n_tok:])):
                if any(math.isnan(x) for x in part):
                    raise TdtContractError("a logit is NaN, so the lattice cannot be scored",
                                           batch=batch, row=i, col=u, slice=label)
                if not any(math.isfinite(x) for x in part):
                    raise TdtContractError(
                        "this slice carries no finite logit, so its log-softmax is undefined: a "
                        "malformed input, not an unreachable alignment",
                        batch=batch, row=i, col=u, slice=label)
            hit = (log_softmax(raw[:n_tok]), log_softmax(raw[n_tok:]))
            cache[key] = hit
        return hit

    return at


def alpha_table(T: int, U: int, spec: TdtSpec, at, target_ids, *,
                mutate_allow_blank_zero: bool = False,
                mutate_token_requires_positive: bool = False,
                mutate_forbid_closing_blank: bool = False,
                mutate_only_blank_closes: bool = False,
                mutate_duration_index_offset: int = 0,
                reachable: Callable[[int, int], bool] | None = None) -> list[list[float]]:
    """The Eq. 6 forward recursion, dense and unbanded, in the log domain.

    Iteration order is increasing ``i + u``, which is topological: a blank advances ``i`` by
    at least one, a token advances ``u`` by one, so ``i + u`` strictly increases along any
    path and no state is read before it is written.
    """
    alpha = [[NEG_INF] * (U + 1) for _ in range(T + 1)]
    alpha[0][0] = 0.0
    nd = spec.n_durations
    offset = mutate_duration_index_offset
    for s in range(1, T + U + 1):
        for i in range(min(s, T) + 1):
            u = s - i
            if u < 0 or u > U:
                continue
            if i == T and u != U:
                continue                                   # sentinel is admissible at u == U only
            if reachable is not None and not reachable(i, u):
                continue
            terms: list[float] = []
            for k, d in enumerate(spec.durations):
                if d == 0 and not mutate_allow_blank_zero:
                    continue                               # Eq. 6: blanks over D excluding 0
                src = (i - d, u)
                if src[0] < 0 or src == (i, u) or alpha[src[0]][src[1]] == NEG_INF:
                    continue
                if mutate_forbid_closing_blank and i == T and u == U:
                    continue
                tok_lp, dur_lp = at(src[0], src[1])
                terms.append(alpha[src[0]][src[1]] + tok_lp[spec.blank_index] + dur_lp[(k + offset) % nd])
            if u >= 1:
                wanted = int(target_ids[u - 1])
                for k, d in enumerate(spec.durations):
                    if d == 0 and mutate_token_requires_positive:
                        continue
                    src = (i - d, u - 1)
                    if src[0] < 0 or alpha[src[0]][src[1]] == NEG_INF:
                        continue
                    if mutate_only_blank_closes and i == T and u == U:
                        continue
                    tok_lp, dur_lp = at(src[0], src[1])
                    terms.append(alpha[src[0]][src[1]] + tok_lp[wanted] + dur_lp[(k + offset) % nd])
            if terms:
                alpha[i][u] = logsumexp(terms)
    return alpha


def loss(logits, targets, lengths, spec: TdtSpec, *,
         expected_positions=None, **mut) -> list[float]:
    """Per-utterance negative log-likelihood ``-log alpha[T, U]``.

    ``spec`` is required, with no default: inheriting one would silently choose a duration alphabet
    and a blank id that nobody declared, and the vocabulary is otherwise inferred from the tensor.

    Raises :class:`TdtAlignmentImpossible` rather than returning ``inf`` or ``NaN`` when no
    admissible path exists: an impossible alignment is a contract breach between encoder
    length and target length, not a number to average.

    ``expected_positions`` is the seam. When supplied - per row, produced by
    ``nativevoice.modeling.masking.position_lengths_from_audio`` from the sample counts, the frozen
    frontend and the declared ``StackSpec`` - any ``T_b`` that differs is refused before a single
    lattice cell is computed, which catches the under-declaration no inspection of ``logits`` can
    reveal. It is keyword-only on purpose: a positional spelling would be swallowed by ``**mut``
    and forwarded into the recursion. ``None`` keeps today's behaviour and exists only so the
    arithmetic and oracle suites can keep calling this function directly; every scoring integration
    under ``src/`` or ``tools/`` is expected to supply it, and
    ``tests/test_tdt_length_seam.py`` guards against one that does not.
    """
    batch = validate(logits, targets, lengths, spec)
    if expected_positions is not None:
        check_expected_positions(lengths, expected_positions, batch=batch)
    terminal_T = int(mut.pop("mutate_terminal_column_T", False))
    out: list[float] = []
    for b in range(batch):
        T, U = int(lengths[b][0]), int(lengths[b][1])
        at = _slices_for(logits[b], spec, b)
        table = alpha_table(T, U, spec, at, targets[b], **mut)
        row = T - terminal_T
        value = table[row][U] if 0 <= row <= T else NEG_INF
        if value == NEG_INF:
            raise TdtAlignmentImpossible(
                "no admissible alignment exists for this length and target",
                batch=b, T=T, U=U, durations=list(spec.durations),
                reason="terminal cell unreachable; check that durations include 0 when U > T",
            )
        if not math.isfinite(value):
            raise TdtAlignmentImpossible(
                "the lattice produced a non-finite likelihood", batch=b, T=T, U=U, value=value
            )
        out.append(-value)
    return out


def is_feasible(T: int, U: int, spec: TdtSpec) -> bool:
    """Whether some admissible path exists - decided by reachability, not by a hand-derived bound.

    GEOMETRY ONLY. This predicate reads ``(T, U, durations)`` and nothing else: it never sees a
    logits tensor, so ``True`` does not establish that a caller holds a valid input, and it is
    deliberately able to disagree with :func:`validate`. ``is_feasible(0, 0, spec)`` is ``True``
    because zero steps reach the terminal state, while ``validate`` refuses ``T == 0`` because a
    caller with no encoder position has nothing to score. Use it to ask whether a length pair is
    representable, never to ask whether a batch is well-formed.
    """
    reach = {(0, 0)}
    if (T, U) in reach:
        return True
    for _ in range(T + U + 2):
        nxt = set(reach)
        for (i, u) in reach:
            for d in spec.positive_durations:
                if i + d <= T:
                    nxt.add((i + d, u))
            if u < U:
                for d in spec.durations:
                    if i + d <= T:
                        nxt.add((i + d, u + 1))
        if (T, U) in nxt:
            return True
        if nxt == reach:
            return False
        reach = nxt
    return False
