"""Specification-level TDT lattice and decoder, deliberately written differently from
``tdt_lattice_oracle.py`` (recursion here, enumeration there).

This is NOT the production training loss. It exists so that the tests have two
independently authored implementations to compare, and so that the decoder contract can
be exercised before any trainer exists.

Mutation flags exist only so the test suite can prove it would notice a wrong
implementation. They default to off and are never enabled by the production path.
"""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from tdt_lattice_oracle import BLANK, V3


# --------------------------------------------------------------------------- lattice


def forward_alpha(T, U, Pt, Pd, D, conv=V3, *, allow_blank_zero=False,
                  token_requires_positive=False, terminal_column_T=False):
    """Eq. 6 forward recursion over 1-based ``(t, u)``, exact rationals, log-free.

    Iteration order is increasing ``t + u``, which is the topological order because every
    transition raises ``t + u``: a blank by ``d >= 1``, a token by ``1`` in ``u``.
    """
    last = T + 1 if not terminal_column_T else T
    a = {}
    for order in range(1, last + U + 1):
        for t in range(1, last + 2):
            u = order - t
            if u < 0 or u > U:
                continue
            if (t, u) == (1, 0):
                a[(t, u)] = Fraction(1)
                continue
            if conv is V3 and t == T + 1 and u != U:
                a[(t, u)] = Fraction(0)
                continue
            total = Fraction(0)
            for d in D:
                if d == 0 and not allow_blank_zero:
                    continue
                src = (t - d, u)
                if src in a and conv.is_source(src[0], T):
                    total += a[src] * Fraction(Pt[src][BLANK]) * Fraction(Pd[src][d])
            if u >= 1:
                for d in D:
                    if d == 0 and token_requires_positive:
                        continue
                    src = (t - d, u - 1)
                    if src in a and conv.is_source(src[0], T):
                        total += a[src] * Fraction(Pt[src][f"y{u}"]) * Fraction(Pd[src][d])
            a[(t, u)] = total
    if terminal_column_T:
        return a, (T, U)
    return a, (T + 1, U)


def p_tdt(T, U, Pt, Pd, D, conv=V3, **mut):
    a, terminal = forward_alpha(T, U, Pt, Pd, D, conv, **mut)
    return a.get(terminal, Fraction(0))


def backward_alpha_free(T, U, Pt, Pd, D, conv=V3):
    """Eq. 8 backward recursion, written independently of the oracle's enumeration."""
    b = {}
    for order in range(T + 1 + U, 0, -1):
        for t in range(T + 1, 0, -1):
            u = order - t
            if u < 0 or u > U:
                continue
            if (t, u) == (T + 1, U):
                b[(t, u)] = Fraction(1)
                continue
            if conv is V3 and t == T + 1 and u != U:
                b[(t, u)] = Fraction(0)
                continue
            if not conv.is_source(t, T):
                b[(t, u)] = Fraction(0)
                continue
            s = (t, u)
            total = Fraction(0)
            for d in D:
                if d == 0:
                    continue
                tgt = (t + d, u)
                if tgt in b and conv.is_target(tgt[0], tgt[1], T, U):
                    total += Fraction(Pt[s][BLANK]) * Fraction(Pd[s][d]) * b[tgt]
            if u < U:
                for d in D:
                    tgt = (t + d, u + 1)
                    if tgt in b and conv.is_target(tgt[0], tgt[1], T, U):
                        total += Fraction(Pt[s][f"y{u + 1}"]) * Fraction(Pd[s][d]) * b[tgt]
            b[(t, u)] = total
    return b


def cut_total(T, U, Pt, Pd, D, n, conv=V3):
    """Eq. 19 plus Eq. 23's jump terms, at diagonal ``n``.

    alpha from the recursion, beta from the recursion, each of which the tests compare
    against the enumeration oracle separately.
    """
    a, _ = forward_alpha(T, U, Pt, Pd, D, conv)
    b = backward_alpha_free(T, U, Pt, Pd, D, conv)
    on_diag = sum((a[(t, u)] * b[(t, u)]
                   for t in range(1, T + 2) for u in range(U + 1)
                   if (t, u) in a and (t, u) in b and t + u == n), Fraction(0))
    crossing = Fraction(0)
    for t in range(1, T + 2):
        for u in range(U + 1):
            s = (t, u)
            if s not in a or a[s] == 0 or t + u >= n or not conv.is_source(t, T):
                continue
            for d in D:
                if d == 0:
                    continue
                tgt = (t + d, u)
                if tgt[0] <= T + 1 and tgt[0] + tgt[1] > n and tgt in b:
                    crossing += a[s] * Fraction(Pt[s][BLANK]) * Fraction(Pd[s][d]) * b[tgt]
            if u < U:
                for d in D:
                    tgt = (t + d, u + 1)
                    if tgt[0] <= T + 1 and tgt[0] + tgt[1] > n and tgt in b:
                        crossing += a[s] * Fraction(Pt[s][f"y{u + 1}"]) * Fraction(Pd[s][d]) * b[tgt]
    return on_diag + crossing


# --------------------------------------------------------------------------- decoder


class Status(str, Enum):
    COMPLETE = "COMPLETE"
    END_OF_INPUT_UNCONSUMED = "END_OF_INPUT_UNCONSUMED"
    OVERSHOOT_ZERO_LIKELIHOOD = "OVERSHOOT_ZERO_LIKELIHOOD"
    NO_POSITIVE_DURATION_AVAILABLE = "NO_POSITIVE_DURATION_AVAILABLE"
    STALL_MAX_ITERATIONS = "STALL_MAX_ITERATIONS"
    OUTPUT_LENGTH_LIMIT_REACHED = "OUTPUT_LENGTH_LIMIT_REACHED"


class IncompleteDecoding(Exception):
    """Raised when a caller asks for a transcript the decoder did not legitimately earn."""


@dataclass
class Outcome:
    status: Status
    hypothesis: tuple = ()
    trace: tuple = ()
    position: int = 0
    steps: int = 0
    zero_duration_tokens: int = 0
    guard_fired: int = 0
    cap_fired: int = 0
    limit_fired: int = 0

    def transcript(self):
        """Only COMPLETE may be treated as a successful transcript."""
        if self.status is not Status.COMPLETE:
            raise IncompleteDecoding(
                f"refusing to return a transcript: status={self.status.value} "
                f"emitted={len(self.hypothesis)} position={self.position}"
            )
        return self.hypothesis


def decode(T, U, Pt, Pd, D, *, progress_obligation=True, terminal_validation=True,
           max_symbols_per_step=None, max_iterations=None, output_length_limit=None):
    """Greedy TDT decoding with an explicit, testable contract.

    Training-lattice facts (Eq. 6): a step that appends no token must advance by a
    positive duration; a token may take any duration in D, including 0; a path lands on
    the terminal node with durations summing to exactly T.

    Inference-only policies, none of them implied by the recurrence:
    ``max_symbols_per_step``, ``max_iterations``, ``output_length_limit``.
    """
    positive = [d for d in D if d > 0]
    if progress_obligation and not positive:
        return Outcome(Status.NO_POSITIVE_DURATION_AVAILABLE, position=1)
    budget = max_iterations if max_iterations is not None else T + U + 2
    hyp, trace, t, steps = [], [], 1, 0
    zero_dur, guard_fired, cap_fired, limit_fired = 0, 0, 0, 0
    per_frame = {}

    while t < T + 1:
        steps += 1
        if steps > budget:
            return Outcome(Status.STALL_MAX_ITERATIONS, tuple(hyp), tuple(trace), t, steps,
                           zero_dur, guard_fired, cap_fired, limit_fired)
        u = len(hyp)
        s = (t, u)
        if s not in Pt:
            return Outcome(Status.OVERSHOOT_ZERO_LIKELIHOOD, tuple(hyp), tuple(trace), t, steps,
                           zero_dur, guard_fired, cap_fired, limit_fired)
        token = max(Pt[s].items(), key=lambda kv: kv[1])[0]
        duration = max(Pd[s].items(), key=lambda kv: kv[1])[0]

        emitted_here = per_frame.get(t, 0)
        capped = max_symbols_per_step is not None and emitted_here >= max_symbols_per_step
        if capped:
            cap_fired += 1
        limited = output_length_limit is not None and u >= output_length_limit
        append = token != BLANK and u < U and not capped and not limited
        if limited and token != BLANK:
            limit_fired += 1

        if not append and duration == 0:
            if progress_obligation:
                duration = min(positive)
                guard_fired += 1

        if append:
            hyp.append(token)
            if duration == 0:
                zero_dur += 1
        per_frame[t] = emitted_here + 1
        trace.append((token, duration, append))
        t += duration

    if output_length_limit is not None and len(hyp) < U and limit_fired:
        return Outcome(Status.OUTPUT_LENGTH_LIMIT_REACHED, tuple(hyp), tuple(trace), t, steps,
                       zero_dur, guard_fired, cap_fired, limit_fired)
    if len(hyp) != U:
        return Outcome(Status.END_OF_INPUT_UNCONSUMED, tuple(hyp), tuple(trace), t, steps,
                       zero_dur, guard_fired, cap_fired, limit_fired)
    if terminal_validation and t != T + 1:
        return Outcome(Status.OVERSHOOT_ZERO_LIKELIHOOD, tuple(hyp), tuple(trace), t, steps,
                       zero_dur, guard_fired, cap_fired, limit_fired)
    return Outcome(Status.COMPLETE, tuple(hyp), tuple(trace), t, steps,
                   zero_dur, guard_fired, cap_fired, limit_fired)
