"""Independent oracle for the Token-and-Duration Transducer lattice.

Written to be structurally different from ``tdt_spec.py``: this module never uses a
recursion. Every quantity here is computed by enumerating emission sequences, so a bug
in the forward recurrence cannot hide by agreeing with itself.

Lattice convention (see the report for the source evidence). States are ``(t, u)`` with
``t`` the audio position, 1-based, and ``u`` the number of targets already emitted.
The terminal node is ``(T + 1, U)``. Two rejected variants are selectable so the suite can
demonstrate *why* the default is the one the paper's boundary sentences support:

``sentinel_state_is_source``
    V2: the column ``t = T + 1`` carries an encoder state and can be emitted from.
``terminal_targets_any_u``
    V1: transitions may land on ``(T + 1, u)`` for ``u != U`` (dead ends).

V3 is both flags False.

CONVENTION STATUS - read before trusting the numbers here.

VERIFIED from the paper (Xu et al., ICML 2023, arXiv:2304.06795), quoted:
    "with the base condition beta(T+1,U)=1";
    "In the recursion, we have beta(T+1,u)=alpha(T+1,u)=0, for all u != U";
    alpha(1,0)=1;  P_TDT(y|x)=alpha(T+1,U)  (Eq. 7);  L_TDT = -log P_TDT(y|x)  (Eq. 9);
    the forward sum takes blanks over D excluding 0 and tokens over all of D (Eq. 6);
    the simulated joint in Appendix D is T x U x (V+1+Nd), with T=70, U=10, V=5.

DERIVED, not stated: that the node T+1 carries no encoder state - i.e. T is the number of
encoder output positions and T+1 is a terminal sentinel. This is what the quoted boundary
sentences and the Appendix D joint shape jointly imply, and it is the V3 used below.

UNRESOLVED: the paper never states what T counts. Six retrieval routes failed to produce a
definition sentence (arXiv HTML 404; raw/resolve config 401/403; two signed-proxy fetches 403;
three ar5iv prompts returned "NO SENTENCE DEFINING T WAS FOUND"). One extraction glossed T as
"frames covered by the emitted token", which describes the duration, not T, and is rejected as
evidence. Treat V3 as a documented interpretation with a named open item, never as an
author-verified fact.
"""

from fractions import Fraction

BLANK = "Ø"


class Convention:
    def __init__(self, sentinel_state_is_source=False, terminal_targets_any_u=False):
        self.sentinel_state_is_source = sentinel_state_is_source
        self.terminal_targets_any_u = terminal_targets_any_u

    @property
    def source_limit(self):
        return "T+1" if self.sentinel_state_is_source else "T"

    def is_source(self, t, T):
        return t >= 1 and (t <= T + 1 if self.sentinel_state_is_source else t <= T)

    def is_target(self, t, u, T, U):
        if t < 1 or u < 0 or u > U or t > T + 1:
            return False
        if t == T + 1 and u != U:
            return self.terminal_targets_any_u or self.sentinel_state_is_source
        return True


V3 = Convention()
V2 = Convention(sentinel_state_is_source=True)
V1 = Convention(terminal_targets_any_u=True)


def enumerate_paths(T, U, Pt, Pd, D, conv=V3):
    """All admissible emission sequences from (1, 0) to (T+1, U), with exact weights.

    ``Pt[(t, u)][v]`` and ``Pd[(t, u)][d]`` are the two factorised distributions of Eq. 5.
    Blanks never take duration 0; tokens may take any duration in ``D``.
    """
    found = []

    def walk(t, u, weight, seq):
        if (t, u) == (T + 1, U):
            found.append((weight, tuple(seq)))
            return
        if not conv.is_source(t, T):
            return
        s = (t, u)
        for d in D:
            if d == 0:
                continue
            if conv.is_target(t + d, u, T, U):
                walk(t + d, u, weight * Fraction(Pt[s][BLANK]) * Fraction(Pd[s][d]),
                     seq + ((BLANK, d),))
        if u < U:
            token = f"y{u + 1}"
            for d in D:
                if conv.is_target(t + d, u + 1, T, U):
                    walk(t + d, u + 1, weight * Fraction(Pt[s][token]) * Fraction(Pd[s][d]),
                         seq + ((token, d),))

    walk(1, 0, Fraction(1), ())
    return found


def total_probability(T, U, Pt, Pd, D, conv=V3):
    """p(y|x) by summation over enumerated paths. No recursion involved."""
    return sum((w for w, _ in enumerate_paths(T, U, Pt, Pd, D, conv)), Fraction(0))


def completion_probability(T, U, Pt, Pd, D, state, conv=V3):
    """beta(state) by enumeration: probability of reaching the terminal node from ``state``."""
    t0, u0 = state
    if (t0, u0) == (T + 1, U):
        return Fraction(1)
    found = []

    def walk(t, u, weight, depth):
        if (t, u) == (T + 1, U):
            found.append(weight)
            return
        if not conv.is_source(t, T) or depth > T + U + 4:
            return
        s = (t, u)
        for d in D:
            if d == 0:
                continue
            if conv.is_target(t + d, u, T, U):
                walk(t + d, u, weight * Fraction(Pt[s][BLANK]) * Fraction(Pd[s][d]), depth + 1)
        if u < U:
            token = f"y{u + 1}"
            for d in D:
                if conv.is_target(t + d, u + 1, T, U):
                    walk(t + d, u + 1, weight * Fraction(Pt[s][token]) * Fraction(Pd[s][d]), depth + 1)

    walk(t0, u0, Fraction(1), 0)
    return sum(found, Fraction(0))


def is_admissible(T, U, D, trace, conv=V3):
    """Check an emission trace against the lattice, independently of how it was produced.

    Accepts ``(label, duration)`` from the enumeration oracle, or
    ``(label, duration, appended)`` from the decoder, where ``appended`` records whether a
    token actually entered the hypothesis. A step that appends nothing — a blank, a capped
    emission, or an emission at the output limit — must advance the audio position.

    Requirements: every non-appending step has duration in ``D`` and greater than 0; every
    appending step has duration in ``D``; the durations sum to exactly ``T``; exactly ``U``
    tokens were appended.
    """
    u = 0
    total = 0
    for step in trace:
        if len(step) == 3:
            v, d, appended = step
        else:
            v, d = step
            appended = (v != BLANK)
        if d not in D:
            return False
        if not appended:
            if d == 0:
                return False
        elif v == BLANK:
            return False
        else:
            u += 1
        total += d
    return u == U and total == T


def random_tables(T, U, D, seed):
    """Deterministic pseudo-probabilities as exact rationals, shared by both modules."""
    import random

    rnd = random.Random(seed)
    Pt, Pd = {}, {}
    names = [BLANK] + [f"y{i + 1}" for i in range(U)]
    for t in range(1, T + 3):
        for u in range(U + 1):
            raw = [rnd.random() + 0.05 for _ in names]
            s = sum(raw)
            Pt[(t, u)] = {n: Fraction(int(round(x / s * 10**9)), 10**9) for n, x in zip(names, raw)}
            raw = [rnd.random() + 0.05 for _ in D]
            s = sum(raw)
            Pd[(t, u)] = {d: Fraction(int(round(x / s * 10**9)), 10**9) for d, x in zip(D, raw)}
    return Pt, Pd
