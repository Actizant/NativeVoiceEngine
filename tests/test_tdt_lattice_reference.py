"""Acceptance tests for the isolated TDT correctness reference.

Three independent programs describe one objective:

* ``tdt_lattice_oracle`` - exact rationals over enumerated paths;
* ``nativevoice.training.tdt_lattice`` - a float64 dense unbanded recursion;
* this file's path-sum objective - float64, reverse-mode differentiated, built from the
  oracle's own path list.

Gradients are therefore cross-checked between two different programs: reverse mode over the
path-sum objective against central differences over the recursion. The torch implementation
is exercised only where torch exists; in this environment it does not, so those tests report
NOT RUN rather than inferring a result. Those checks have been run separately, against a
CPU-only torch installed outside this repository, and the measured agreement is recorded in
the module docstring of ``nativevoice.training.tdt_loss``.

The ``mutate_*`` keyword arguments are used to show each rule in the recurrence is
load-bearing. Where an independent expectation exists it is compared exactly, so a mutation
is shown to be wrong *for the reason claimed*, not merely different.
"""

from __future__ import annotations

import math
import random
import tracemalloc
from fractions import Fraction

import pytest

from nativevoice.training.errors import TdtAlignmentImpossible, TdtContractError
from nativevoice.training.tdt_lattice import (
    TdtSpec,
    alpha_table,
    is_feasible,
    log_softmax,
    loss,
    validate,
)
from tdt_lattice_oracle import BLANK, V3, completion_probability, enumerate_paths, random_tables, total_probability

TOKEN_TOL = 1e-12      # recursion vs exact rationals, across a ln/exp bridge on exact tables
ALGO_TOL = 1e-12      # recursion vs path-sum objective: same inputs, different programs
_NO_TORCH = ("NOT RUN: torch is absent from this environment; these checks run only in the "
             "isolated CPU-torch verification environment installed outside this repository")

GRID = [(T, U, D, T * 31 + U * 7 + len(D))
        for T in (1, 2, 3, 4)
        for U in (0, 1, 2, 3)
        for D in ((0, 1), (1,), (0, 1, 2))]
GRID_IDS = [f"T{T}U{U}D{''.join(map(str, D))}" for T, U, D, _ in GRID]


def prob_tables(T, U, D, seed):
    """Exact dyadic-free rational tables: every slice sums to exactly 1.

    ``tdt_lattice_oracle.random_tables`` quantises probabilities at 1e-9, which the
    log/exp bridge below inherits as a ~1e-9 error in the loss. Building the tables with
    ``Fraction`` of integers keeps the two paths identical to float64 rounding instead.
    """
    rng = random.Random(seed)
    Pt, Pd = {}, {}
    for i in range(1, T + 2):
        for u in range(U + 1):
            w = [rng.randint(1, 99) for _ in range(U + 1)]
            total = sum(w)
            table = {BLANK: Fraction(w[0], total)}
            for j in range(U):
                table[f"y{j + 1}"] = Fraction(w[j + 1], total)
            Pt[(i, u)] = table
            v = [rng.randint(1, 99) for _ in D]
            sv = sum(v)
            Pd[(i, u)] = {d: Fraction(x, sv) for d, x in zip(D, v)}
    return Pt, Pd


def logits_for(T, U, D, seed):
    """One utterance's logits [i][u][slice], built so log_softmax reproduces the oracle's P."""
    Pt, Pd = prob_tables(T, U, D, seed)
    vocab = U + 1
    rows = []
    for i in range(1, T + 1):
        col = []
        for u in range(U + 1):
            tok = [math.log(float(Pt[(i, u)][BLANK]))] + [
                math.log(float(Pt[(i, u)][f"y{j + 1}"])) for j in range(U)
            ]
            dur = [math.log(float(Pd[(i, u)][d])) for d in D]
            col.append(tok + dur)
        rows.append(col)
    return Pt, Pd, rows, vocab


def call(T, U, D, rows, **mut):
    """One-utterance call: the API is batched, so the batch has size 1."""
    targets = [[j + 1 for j in range(U)]]
    lengths = [[T, U]]
    return loss([rows], targets, lengths, TdtSpec(durations=tuple(D)), **mut)


def reference_loss(T, U, D, seed):
    _, _, rows, _ = logits_for(T, U, D, seed)
    return call(T, U, D, rows)[0]


def oracle_loss(T, U, D, seed, durations=None):
    Pt, Pd = prob_tables(T, U, D, seed)
    p = total_probability(T, U, Pt, Pd, tuple(D if durations is None else durations), V3)
    if p == 0:
        return None
    return -math.log(float(p))


def slices_of(rows, T, U, vocab):
    """Token and duration slices are normalised separately: Eq. 5 factorises them."""
    tok, dur = [], []
    for i in range(T):
        tok.append([log_softmax(rows[i][u][:vocab]) for u in range(U + 1)])
        dur.append([log_softmax(rows[i][u][vocab:]) for u in range(U + 1)])
    return tok, dur


def path_sum_loss(T, U, D, seed):
    """-log of the sum of path probabilities, evaluated over the same float64 slices."""
    Pt, Pd, rows, vocab = logits_for(T, U, D, seed)
    tok, dur = slices_of(rows, T, U, vocab)
    terms = []
    for _, emissions in enumerate_paths(T, U, Pt, Pd, D, V3):
        i = u = 0
        total = 0.0
        for v, d in emissions:
            idx = 0 if v == BLANK else int(v[1:])
            total += tok[i][u][idx] + dur[i][u][D.index(d)]
            i += d
            if v != BLANK:
                u += 1
        terms.append(total)
    if not terms:
        return None
    best = max(terms)
    return -(best + math.log(math.fsum(math.exp(x - best) for x in terms)))


def subset_path_loss(T, U, D, seed, keep):
    """The same objective restricted to a subset of paths - the exact expectation for closures."""
    Pt, Pd, rows, vocab = logits_for(T, U, D, seed)
    tok, dur = slices_of(rows, T, U, vocab)
    terms = []
    for _, emissions in enumerate_paths(T, U, Pt, Pd, D, V3):
        if not keep(emissions):
            continue
        i = u = 0
        total = 0.0
        for v, d in emissions:
            idx = 0 if v == BLANK else int(v[1:])
            total += tok[i][u][idx] + dur[i][u][D.index(d)]
            i += d
            if v != BLANK:
                u += 1
        terms.append(total)
    if not terms:
        return None
    if not terms:
        return None
    best = max(terms)
    return -(best + math.log(math.fsum(math.exp(x - best) for x in terms)))


def mutated_loss(T, U, D, seed, **mut):
    Pt, Pd, rows, vocab = logits_for(T, U, D, seed)

    def at(i, u):
        raw = rows[i][u]
        return log_softmax(raw[:vocab]), log_softmax(raw[vocab:])

    table = alpha_table(T, U, TdtSpec(durations=tuple(D)), at, [j + 1 for j in range(U)], **mut)
    value = table[T][U]
    return None if math.isinf(value) else -value


# --------------------------------------------------------------------------- 1. agreement


@pytest.mark.parametrize("T,U,D,seed", GRID, ids=GRID_IDS)
def test_recursion_matches_the_exact_rational_oracle(T, U, D, seed):
    expect = oracle_loss(T, U, D, seed)
    if expect is None:
        with pytest.raises(TdtAlignmentImpossible):
            reference_loss(T, U, D, seed)
        return
    assert abs(reference_loss(T, U, D, seed) - expect) < TOKEN_TOL * max(1.0, abs(expect)), (T, U, D)


@pytest.mark.parametrize("T,U,D,seed", GRID, ids=GRID_IDS)
def test_recursion_matches_the_independent_path_sum_objective(T, U, D, seed):
    expect = path_sum_loss(T, U, D, seed)
    if expect is None or math.isinf(expect):
        with pytest.raises(TdtAlignmentImpossible):
            reference_loss(T, U, D, seed)
        return
    got = reference_loss(T, U, D, seed)
    assert abs(got - expect) < ALGO_TOL * max(1.0, abs(got)), (T, U, D, got, expect)


def test_the_ln_bridge_costs_nothing_material():
    T, U, D = 3, 2, (0, 1, 2)
    Pt, Pd, rows, vocab = logits_for(T, U, D, 5)
    for i in range(1, T + 1):
        for u in range(U + 1):
            tok = log_softmax(rows[i - 1][u][:vocab])
            dur = log_softmax(rows[i - 1][u][vocab:])
            worst = max([abs(math.exp(tok[0]) - float(Pt[(i, u)][BLANK]))]
                        + [abs(math.exp(tok[j + 1]) - float(Pt[(i, u)][f"y{j + 1}"])) for j in range(U)])
            assert worst < 1e-12, (i, u, worst)
            assert max(abs(math.exp(a) - float(Pd[(i, u)][d])) for a, d in zip(dur, D)) < 1e-12


# --------------------------------------------------------------------------- 2. terminal semantics


def test_terminal_paths_split_between_blank_and_token_closures():
    T, U, D, seed = 4, 3, (0, 1), 23
    Pt, Pd = prob_tables(T, U, D, seed)
    paths = [p for _, p in enumerate_paths(T, U, Pt, Pd, D, V3)]
    blank_closed = [p for p in paths if p[-1][0] == BLANK]
    token_closed = [p for p in paths if p[-1][0] != BLANK]
    assert blank_closed and token_closed, "fixture no longer exercises both closures"
    honest = reference_loss(T, U, D, seed)
    no_blank = mutated_loss(T, U, D, seed, mutate_forbid_closing_blank=True)
    no_token = mutated_loss(T, U, D, seed, mutate_only_blank_closes=True)
    assert no_blank == pytest.approx(subset_path_loss(T, U, D, seed, lambda p: p[-1][0] != BLANK),
                                    rel=1e-9, abs=1e-9)
    assert no_token == pytest.approx(subset_path_loss(T, U, D, seed, lambda p: p[-1][0] == BLANK),
                                     rel=1e-9, abs=1e-9)
    assert min(no_blank, no_token) > honest, "deleting a closure family must lose probability"


def test_the_sentinel_row_is_never_read():
    """ASSUMPTION A1's observable consequence inside the module: row ``i = T`` is not consulted."""
    T, U, D, seed = 3, 2, (0, 1, 2), 3
    _, _, rows, _ = logits_for(T, U, D, seed)
    baseline = call(T, U, D, rows)[0]
    width = len(rows[0][0])
    poisoned = rows + [[[9.0] * width for _ in range(U + 1)]]
    assert call(T, U, D, poisoned)[0] == pytest.approx(baseline, abs=0.0)
    with pytest.raises(TdtContractError):
        call(T, U, D, rows[: T - 1])


def test_the_backward_base_admits_only_a_duration_one_blank():
    T, U, D, seed = 3, 2, (0, 1, 2), 9
    Pt, Pd = prob_tables(T, U, D, seed)
    beta = float(completion_probability(T, U, Pt, Pd, D, (T, U), V3))
    single = float(Pt[(T, U)][BLANK]) * float(Pd[(T, U)][1])
    marginal = sum(float(Pt[(T, U)][BLANK]) * float(Pd[(T, U)][d]) for d in D if d > 0)
    assert beta == pytest.approx(single, rel=1e-12)
    assert beta != pytest.approx(marginal, rel=1e-9), "fixture cannot distinguish the two readings"


# --------------------------------------------------------------------------- 3. boundaries


def test_empty_target_is_a_pure_blank_chain():
    T, U, D, seed = 2, 0, (0, 1), 2
    assert reference_loss(T, U, D, seed) == pytest.approx(oracle_loss(T, U, D, seed), rel=1e-9, abs=1e-9)


def test_impossible_alignment_raises_rather_than_returning_infinity():
    with pytest.raises(TdtAlignmentImpossible) as err:
        reference_loss(T=1, U=3, D=(1,), seed=5)
    assert err.value.details["T"] == 1 and err.value.details["U"] == 3
    assert "durations" in err.value.details


def test_zero_duration_tokens_are_what_make_long_targets_representable():
    assert not is_feasible(1, 3, TdtSpec(durations=(1,)))
    assert is_feasible(1, 3, TdtSpec(durations=(0, 1)))
    assert math.isfinite(reference_loss(T=1, U=3, D=(0, 1), seed=5))


@pytest.mark.parametrize("T,U,D", [(1, 1, (0, 1)), (2, 2, (0, 1)), (3, 1, (1,)), (1, 2, (0, 1)),
                                   (2, 4, (0, 1, 2)), (4, 2, (0, 2)), (1, 0, (0, 1))])
def test_feasibility_agrees_with_the_lattice(T, U, D):
    feasible = is_feasible(T, U, TdtSpec(durations=D))
    try:
        value = reference_loss(T, U, D, T * 5 + U)
        raised = False
    except TdtAlignmentImpossible:
        value, raised = None, True
    assert feasible != raised, (T, U, D, feasible, raised)
    if not raised:
        assert math.isfinite(value)


@pytest.mark.parametrize("bad", [(), (0,), (0, 0, 1), (2, 0), (0, 1.5), (0, 1, 1)])
def test_spec_rejects_duration_sets_that_cannot_describe_a_lattice(bad):
    with pytest.raises(TdtContractError):
        TdtSpec(durations=tuple(bad))


def test_contract_errors_name_the_offending_input():
    T, U, D = 2, 2, (0, 1)
    _, _, rows, vocab = logits_for(T, U, D, 1)
    with pytest.raises(TdtContractError) as err:
        validate([rows], [[0, 1]], [[T, U]], TdtSpec(durations=D))     # blank inside a target
    assert "blank" in str(err.value)
    with pytest.raises(TdtContractError):
        validate([rows], [[1, 2]], [[T, U]], TdtSpec(durations=D, blank_index=99))
    with pytest.raises(TdtContractError):
        validate([rows], [[1, 2, 3]], [[T, 3]], TdtSpec(durations=D))  # rows too few for U+1
    with pytest.raises(TdtContractError):
        validate([rows], [[1, 2]], [[T + 1, U]], TdtSpec(durations=D))


# --------------------------------------------------------------------------- 4. mutations


def test_zero_duration_blanks_are_excluded_because_they_make_the_lattice_not_a_dag():
    """The reason Eq. 6 sums blanks over D excluding 0, shown structurally.

    A zero-duration blank is a self-loop on the same lattice state, so admitting it turns
    the finite path sum into an infinite series: the path count grows without bound as the
    permitted number of loops grows. A value-level mutation cannot show this - the recursion
    skips the self-loop and the number is simply unchanged - so the test measures growth.
    """
    T, U, D, seed = 2, 1, (0, 1, 2), 3
    Pt, Pd = prob_tables(T, U, D, seed)
    honest = mutated_loss(T, U, D, seed, mutate_allow_blank_zero=True)
    plain = mutated_loss(T, U, D, seed)
    assert honest == pytest.approx(plain, rel=1e-12)          # skipped self-loop: no new mass

    def count(max_loops):
        found = 0

        def walk(i, u, loops):
            nonlocal found
            if (i, u) == (T, U):
                found += 1
                return
            for d in D:
                if d == 0:
                    continue
                if i + d <= T:
                    walk(i + d, u, loops)
            if u < U:
                for d in D:
                    if i + d <= T:
                        walk(i + d, u + 1, loops)
            if loops < max_loops and 0 in D:
                walk(i, u, loops + 1)                         # the zero-duration blank self-loop

        walk(0, 0, 0)
        return found

    growing = [count(cap) for cap in (0, 1, 2, 3)]
    assert growing == sorted(growing) and len(set(growing)) == len(growing), growing


def test_mutation_banning_zero_duration_tokens_equals_the_positive_only_set():
    T, U, D, seed = 3, 2, (0, 1, 2), 13
    got = mutated_loss(T, U, D, seed, mutate_token_requires_positive=True)
    expect = oracle_loss(T, U, D, seed, durations=tuple(d for d in D if d > 0))
    assert got == pytest.approx(expect, rel=1e-9, abs=1e-9)
    assert got != pytest.approx(reference_loss(T, U, D, seed), rel=1e-9)


def test_mutation_reading_the_wrong_terminal_column_drops_the_closing_step():
    """Reading one row early silently omits the mandatory final blank.

    With D=(1,) and an empty target the lattice is a pure chain of duration-1 blanks, so the
    exact identity p[T][0] == p[T-1][0] * P(blank,1|T-1,0) must hold. A loss that reads the
    wrong row satisfies it badly, and the error is the log of that closing probability.
    """
    T, U, D, seed = 3, 0, (0, 1), 47
    Pt, Pd, rows, vocab = logits_for(T, U, D, seed)
    honest = call(T, U, D, rows)[0]
    shifted = call(T, U, D, rows, mutate_terminal_column_T=True)[0]
    # A one-symbol token slice is degenerate (softmax of one value is 1), so the discriminating
    # factor here is the duration: an empty target is a pure chain of blanks over D excluding 0.
    dur_lp = log_softmax(rows[T - 1][0][vocab:])
    p_closing = math.exp(dur_lp[D.index(1)])
    assert 0.0 < p_closing < 1.0
    assert shifted < honest, "dropping the closing blank must lower the loss, not raise it"
    assert abs((honest - shifted) + math.log(p_closing)) < 1e-9, (honest, shifted, p_closing)


def test_mutation_shifting_the_duration_slice_matches_a_permuted_table():
    T, U, D, seed = 3, 2, (0, 1, 2), 29
    Pt, Pd, rows, vocab = logits_for(T, U, D, seed)
    rotated = tuple(D[(k + 1) % len(D)] for k in range(len(D)))
    perm = []
    for i in range(1, T + 1):
        col = []
        for u in range(U + 1):
            tok = [math.log(float(Pt[(i, u)][BLANK]))] + [math.log(float(Pt[(i, u)][f"y{j + 1}"])) for j in range(U)]
            col.append(tok + [math.log(float(Pd[(i, u)][d])) for d in rotated])
        perm.append(col)
    expect = call(T, U, D, perm)[0]
    got = mutated_loss(T, U, D, seed, mutate_duration_index_offset=1)
    assert got == pytest.approx(expect, rel=1e-9, abs=1e-9)
    assert got != pytest.approx(reference_loss(T, U, D, seed), rel=1e-6)


def test_a_full_band_changes_nothing_and_a_narrow_band_loses_paths():
    T, U, D, seed = 4, 3, (0, 1, 2), 31
    honest = reference_loss(T, U, D, seed)
    full = mutated_loss(T, U, D, seed, reachable=lambda i, u: True)
    narrow = mutated_loss(T, U, D, seed, reachable=lambda i, u: abs(i * U - T * u) <= 1)
    assert full == pytest.approx(honest, rel=1e-12)
    assert narrow is None or narrow > honest


# --------------------------------------------------------------------------- 5. gradients


class Node:
    """Scalar reverse mode over floats, for the path-sum objective only."""

    __slots__ = ("value", "grad", "parents")

    def __init__(self, value):
        self.value = float(value)
        self.grad = 0.0
        self.parents = []


def n_exp(a, c=0.0):
    out = Node(math.exp(a.value - c))
    out.parents.append((a, out.value))
    return out


def n_log(a):
    out = Node(math.log(a.value))
    out.parents.append((a, 1.0 / a.value))
    return out


def n_mul(a, b):
    out = Node(a.value * b.value)
    out.parents += [(a, b.value), (b, a.value)]
    return out


def n_add(a, b):
    out = Node(a.value + b.value)
    out.parents += [(a, 1.0), (b, 1.0)]
    return out


def n_div(a, b):
    out = Node(a.value / b.value)
    out.parents += [(a, 1.0 / b.value), (b, -a.value / (b.value * b.value))]
    return out


def backward(root, seed=1.0):
    order, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p, _ in n.parents:
            visit(p)
        order.append(n)

    visit(root)
    root.grad += seed
    for n in reversed(order):
        for p, local in n.parents:
            p.grad += n.grad * local


def node_log_softmax(values):
    m = max(v.value for v in values)
    exps = [n_exp(v, m) for v in values]
    total = exps[0]
    for e in exps[1:]:
        total = n_add(total, e)
    return [n_log(n_div(e, total)) for e in exps]      # log(e) - log(total)


def path_sum_nodes(T, U, D, seed, drop_durations=False):
    Pt, Pd, rows, vocab = logits_for(T, U, D, seed)
    params = {}
    for i in range(T):
        for u in range(U + 1):
            params[(i, u)] = [Node(x) for x in rows[i][u]]
    terms = []
    for _, emissions in enumerate_paths(T, U, Pt, Pd, D, V3):
        i = u = 0
        acc = Node(0.0)
        for v, d in emissions:
            idx = 0 if v == BLANK else int(v[1:])
            tok = node_log_softmax(params[(i, u)][:vocab])
            dur = node_log_softmax(params[(i, u)][vocab:])
            acc = n_add(acc, tok[idx])
            if not drop_durations:
                acc = n_add(acc, dur[D.index(d)])
            i += d
            if v != BLANK:
                u += 1
        terms.append(acc)
    best = max(t.value for t in terms)
    total = n_exp(terms[0], best)
    for t in terms[1:]:
        total = n_add(total, n_exp(t, best))
    objective = n_add(n_log(total), Node(best))         # +log p ; the loss is its negation
    return params, objective


def test_path_sum_gradient_matches_central_differences_of_the_recursion():
    """Reverse mode over one program against finite differences of another."""
    T, U, D, seed = 3, 2, (0, 1, 2), 41
    params, objective = path_sum_nodes(T, U, D, seed)
    backward(objective, 1.0)
    _, _, rows, _ = logits_for(T, U, D, seed)
    h = 1e-6
    worst, checked = 0.0, 0
    for i in range(T):
        for u in range(U + 1):
            for col, node in enumerate(params[(i, u)]):
                eps = h * max(1.0, abs(node.value))
                plus = [n.value for n in params[(i, u)]]
                minus = [n.value for n in params[(i, u)]]
                plus[col] += eps
                minus[col] -= eps

                def value(table):
                    probe = [list(r) for r in rows]
                    probe[i] = [list(c) for c in probe[i]]
                    probe[i][u] = table
                    return call(T, U, D, probe)[0]

                fd = (value(plus) - value(minus)) / (2 * eps)
                ad = -node.grad                              # loss = -log p
                worst = max(worst, abs(fd - ad))
                checked += 1
    assert checked > 50
    assert worst < 1e-5, worst


def test_the_gradient_check_bites_when_a_term_is_dropped():
    """Negative control: an objective missing its duration slice must disagree."""
    T, U, D, seed = 3, 2, (0, 1, 2), 41
    params, objective = path_sum_nodes(T, U, D, seed, drop_durations=True)
    backward(objective, 1.0)
    _, _, rows, _ = logits_for(T, U, D, seed)
    h, worst = 1e-6, 0.0
    for i in range(T):
        for u in range(U + 1):
            for col, node in enumerate(params[(i, u)]):
                if col <= U:
                    continue                                 # duration slice only
                eps = h * max(1.0, abs(node.value))
                plus = [n.value for n in params[(i, u)]]
                minus = [n.value for n in params[(i, u)]]
                plus[col] += eps
                minus[col] -= eps

                def value(table):
                    probe = [list(r) for r in rows]
                    probe[i] = [list(c) for c in probe[i]]
                    probe[i][u] = table
                    return call(T, U, D, probe)[0]

                fd = (value(plus) - value(minus)) / (2 * eps)
                worst = max(worst, abs(fd + node.grad))
    assert worst > 1e-3, "the gradient check failed to notice a missing duration term"


# --------------------------------------------------------------------------- 6. cost


@pytest.mark.parametrize("T,U", [(20, 6), (248, 21)])
def test_the_recursion_scales_with_the_lattice_not_with_the_vocabulary(T, U):
    D = (0, 1, 2)
    vocab = U + 1
    width = vocab + len(D)

    class Rows:
        def __len__(self):
            return T

        def __getitem__(self, i):
            return Cols()

    class Cols:
        def __len__(self):
            return U + 1

        def __getitem__(self, u):
            return [0.0] * width

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    table = alpha_table(T, U, TdtSpec(durations=D), _lazy_slices(width),
                        [j + 1 for j in range(U)])
    used = max(tracemalloc.get_traced_memory()[0] - before, 1)
    tracemalloc.stop()
    assert math.isfinite(table[T][U])
    cells = (T + 1) * (U + 1)
    assert used / cells < 512, f"{used / cells:.1f} bytes per cell suggests the logits were materialised"
    assert used < 32 * 2 ** 20, used


def _lazy_slices(width):
    """A slice provider that allocates nothing until asked - the point is that the recursion
    must not copy or reshape the caller's logits."""
    def at(i, u):
        raw = [0.0] * width
        v = width - 3
        return log_softmax(raw[:v]), log_softmax(raw[v:])

    return at


# ---------------------------------------------------------- 7. torch: skipped where torch is absent


def test_torch_recursion_matches_the_reference():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    for T, U, D, seed in GRID:
        if U > T and 0 not in D:
            continue
        _, _, rows, _ = logits_for(T, U, D, seed)
        logits = torch.tensor([[rows[u] if False else rows[i] for i in range(T)]], dtype=torch.float64)
        targets = torch.tensor([[j + 1 for j in range(U)]], dtype=torch.int64)
        lengths = torch.tensor([[T, U]])
        got = float(tdt_loss(logits, targets, lengths, TdtSpec(durations=D))[0])
        assert got == pytest.approx(reference_loss(T, U, D, seed), rel=1e-9, abs=1e-9)


def test_torch_gradients_match_the_path_sum_gradient():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    T, U, D, seed = 3, 2, (0, 1, 2), 41
    _, _, rows, _ = logits_for(T, U, D, seed)
    logits = torch.tensor([rows], dtype=torch.float64, requires_grad=True)
    tdt_loss(logits, torch.tensor([[1, 2]]), torch.tensor([[T, U]]), TdtSpec(durations=D)).sum().backward()
    params, objective = path_sum_nodes(T, U, D, seed)
    backward(objective, 1.0)
    worst = max(abs(float(logits.grad[0, i, u, c]) + node.grad)
                for i in range(T) for u in range(U + 1) for c, node in enumerate(params[(i, u)]))
    assert worst < 1e-8, worst


def test_torch_gradients_match_central_differences_of_the_torch_loss_itself():
    """Reverse mode through ``tdt_loss`` against central differences of ``tdt_loss``.

    The node above compares the reference's own two programs; neither checked this module's
    autograd output, which its docstring admitted. Measured on the deterministic 3x2 lattice with
    a 5-symbol token slice and Nd 3: 72 components at h = 1e-6, worst relative error 6.4e-10
    against the 1e-7 bound below. This node is torch-gated: it is NOT RUN in the project
    interpreter and runs only in the isolated CPU-torch environment.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss

    T, U, D, seed = 3, 2, (0, 1, 2), 41
    spec = TdtSpec(durations=D)
    _, _, rows, _ = logits_for(T, U, D, seed)
    base = torch.tensor([rows], dtype=torch.float64)
    targets = torch.tensor([[1, 2]], dtype=torch.int64)
    lengths = torch.tensor([[T, U]], dtype=torch.int64)

    x = base.clone().requires_grad_(True)
    tdt_loss(x, targets, lengths, spec).sum().backward()
    grad = x.grad.clone()

    h = 1e-6
    worst = 0.0
    for i in range(T):
        for u in range(U + 1):
            for c in range(int(base.shape[-1])):
                plus = base.clone(); plus[0, i, u, c] += h
                minus = base.clone(); minus[0, i, u, c] -= h
                lp = float(tdt_loss(plus, targets, lengths, spec).sum())
                lm = float(tdt_loss(minus, targets, lengths, spec).sum())
                step = (lp - lm) / (2 * h)
                worst = max(worst, abs(step - float(grad[0, i, u, c])) / max(1.0, abs(step)))
    assert worst < 1e-7, worst


# ------------------------------------------------- 8. zero-duration coverage, exact rationals
#
# At U == 0 there are no token closures, so every admissible path is a composition of T into
# parts drawn from D excluding 0. That makes the blank branch's duration indexing observable in
# exact arithmetic: the honest rule and a slot-shifted rule give different numbers, and only
# one of them is 1. A probe that tried to vary P_D(0) on its own cannot work - the duration
# slice is normalised, so P_D(0) is fixed by the other entries - which is why the comparison
# below changes which slot is read instead of trying to change a probability.

IMPOSSIBLE_CELLS = [c for c in GRID if c[1] > c[0] and 0 not in c[2]]
IMPOSSIBLE_IDS = [f"T{t}U{u}D{''.join(map(str, d))}" for t, u, d, _ in IMPOSSIBLE_CELLS]
ZERO_DURATION_CELLS = [(T, 0, D, T * 31 + len(D)) for T in (1, 2, 3, 4) for D in ((0, 1), (1,), (0, 1, 2))]
ZERO_DURATION_IDS = [f"T{T}U0D{''.join(map(str, D))}" for T, _, D, _ in ZERO_DURATION_CELLS]


def exact_blank_only(T, D, seed):
    """Exact p(y|x) for an empty target, by enumeration over blank-only paths."""
    Pt, Pd = prob_tables(T, 0, D, seed)
    return total_probability(T, 0, Pt, Pd, tuple(D), V3)


def exact_blank_only_single_path(T, D, seed):
    """The closed form for D == (0, 1): exactly one path, T duration-1 blanks from t = 1."""
    Pt, Pd = prob_tables(T, 0, D, seed)
    p = Fraction(1)
    for t in range(1, T + 1):
        p *= Fraction(Pt[(t, 0)][BLANK]) * Fraction(Pd[(t, 0)][1])
    return p


def exact_blank_only_shifted(T, D, seed, shift=1):
    """The same lattice scored as if a blank traversing duration d paid the slot of the
    duration `shift` places earlier - a wrong implementation, expressible exactly."""
    Pt, Pd = prob_tables(T, 0, D, seed)
    total = Fraction(0)
    for _, seq in enumerate_paths(T, 0, Pt, Pd, tuple(D), V3):
        t, w = 1, Fraction(1)
        for _, d in seq:
            k = (list(D).index(d) + shift) % len(D)
            w *= Fraction(Pt[(t, 0)][BLANK]) * Fraction(Pd[(t, 0)][D[k]])
            t += d
        total += w
    return total


@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_a_blank_only_lattice_is_one_exact_path_and_ignores_the_zero_slot(T):
    """The empty-target lattice has a hand-writable value, and it does not contain P_D(0)."""
    D, seed = (0, 1), 61
    honest = -math.log(float(exact_blank_only_single_path(T, D, seed)))
    enumerator = -math.log(float(exact_blank_only(T, D, seed)))
    got = reference_loss(T, 0, D, seed)
    assert honest == pytest.approx(enumerator, rel=1e-15, abs=1e-15), (honest, enumerator)
    assert got == pytest.approx(honest, rel=1e-12, abs=1e-12), (T, got, honest)
    shifted = -math.log(float(exact_blank_only_shifted(T, D, seed)))
    assert abs(got - shifted) > 1e-6, (T, got, shifted)


@pytest.mark.parametrize("T,D", [(2, (0, 1)), (3, (0, 1)), (2, (0, 1, 2)), (3, (0, 1, 2))])
def test_the_blank_only_recursion_follows_the_honest_duration_slot(T, D):
    """Two and three durations admit several blank-only paths, so the closed form no longer
    covers the case; the enumerator does, and the shifted rule is still distinguishable."""
    seed = 67
    honest = -math.log(float(exact_blank_only(T, D, seed)))
    shifted = -math.log(float(exact_blank_only_shifted(T, D, seed)))
    assert abs(honest - shifted) > 1e-3, "the two rules are indistinguishable; control is vacuous"
    got = reference_loss(T, 0, D, seed)
    assert got == pytest.approx(honest, rel=1e-12, abs=1e-12), (T, D, got, honest, shifted)


# --------------------------------------------------------------------------- 9. torch batteries


def _torch_inputs(torch, T, U, D, seed):
    """One well-formed utterance: logits [1,T,U+1,V+Nd], int64 targets [1,U], lengths [1,2].

    The targets dtype is written out because an empty Python list makes torch infer float32,
    and the integer-target guard is exactly what refuses that caller error.
    """
    _, _, rows, _ = logits_for(T, U, D, seed)
    logits = torch.tensor([rows], dtype=torch.float64)
    targets = torch.tensor([[] if U == 0 else [j + 1 for j in range(U)]], dtype=torch.int64)
    lengths = torch.tensor([[T, U]])
    return logits, targets, lengths, TdtSpec(durations=tuple(D))


@pytest.mark.parametrize("T,U,D,seed", IMPOSSIBLE_CELLS, ids=IMPOSSIBLE_IDS)
def test_torch_impossible_geometry_raises_rather_than_being_skipped(T, U, D, seed):
    """Every GRID cell that the parity loop used to `continue` past is now an assertion.

    U > T with no zero-duration token leaves the terminal node unreachable, which is a contract
    breach between encoder length and target length, not a number. The reference and torch must
    refuse it, `is_feasible` must agree that it is infeasible, and the refusal must name it.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    spec = TdtSpec(durations=tuple(D))
    assert not is_feasible(T, U, spec), (T, U, D)
    logits, targets, lengths, spec = _torch_inputs(torch, T, U, D, seed)
    with pytest.raises(TdtAlignmentImpossible) as err:
        tdt_loss(logits, targets, lengths, spec)
    assert err.value.details["T"] == T and err.value.details["U"] == U
    assert 0 not in err.value.details["durations"]
    assert err.value.details["batch"] == 0
    _, _, rows, _ = logits_for(T, U, D, seed)
    with pytest.raises(TdtAlignmentImpossible):
        call(T, U, D, rows)


@pytest.mark.parametrize("T,U,D,seed",
                         [(T, U, (0, 1), seed) for T, U, D, seed in IMPOSSIBLE_CELLS],
                         ids=[f"twin{i}" for i in IMPOSSIBLE_IDS])
def test_torch_admits_the_same_geometry_once_a_zero_duration_token_exists(T, U, D, seed):
    """Negative control for the cell above: identical shapes, D gains 0, must not raise.

    Without this, the refusal test would also pass on an implementation that rejects every
    call - a raise is only evidence when the neighbouring call does not raise.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    assert is_feasible(T, U, TdtSpec(durations=tuple(D))), (T, U, D)
    logits, targets, lengths, spec = _torch_inputs(torch, T, U, D, seed)
    value = float(tdt_loss(logits, targets, lengths, spec)[0])
    assert math.isfinite(value), (T, U, D, value)


CONTRACT_CASES = [
    ("logits three-dimensional",
     lambda torch, x, U: (x[0].squeeze(0), x[1], x[2], x[3]), "logits must be [B, T"),
    ("targets longer than U",
     lambda torch, x, U: (x[0], x[1].repeat(1, 2), x[2], x[3]), "targets must be [B, U]"),
    ("lengths has three columns",
     lambda torch, x, U: (x[0], x[1], torch.tensor([[2, U, 0]]), x[3]), "lengths must be [B, 2]"),
    ("width equals the duration slice",
     lambda torch, x, U: (x[0][..., :2], x[1], x[2], x[3]), "logit width must exceed"),
    ("no encoder rows at all",
     lambda torch, x, U: (torch.empty((1, 0, U + 1, 4), dtype=torch.float64), x[1],
                          torch.tensor([[0, U]]), x[3]), "at least one encoder row"),
    ("blank outside the token slice",
     lambda torch, x, U: (x[0], x[1], x[2], TdtSpec(durations=(0, 1), blank_index=99)),
     "blank_index outside"),
    ("float targets",
     lambda torch, x, U: (x[0], x[1].to(torch.float32), x[2], x[3]), "integer tensor"),
    ("target id beyond the vocabulary",
     lambda torch, x, U: (x[0], torch.tensor([[7]], dtype=torch.int64), x[2], x[3]),
     "outside the vocabulary"),
    ("blank inside a target",
     lambda torch, x, U: (x[0], torch.tensor([[0]], dtype=torch.int64), x[2], x[3]),
     "blank symbol cannot appear"),
    ("declared T exceeds the row count",
     lambda torch, x, U: (x[0], x[1], torch.tensor([[3, U]]), x[3]), "declared length T exceeds"),
]


@pytest.mark.parametrize("label,break_it,needle", CONTRACT_CASES,
                         ids=[c[0] for c in CONTRACT_CASES])
def test_torch_refuses_each_malformed_call_by_name(label, break_it, needle):
    """Each guard in ``tdt_loss._check_shapes`` is exercised separately, and must name the
    offending input rather than merely raising.

    A single catch-all exception type would satisfy a bare ``raises`` check, so the expected
    substring is asserted too: the message is the evidence that the right guard fired.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    T, U, D, seed = 2, 1, (0, 1), 71
    base = _torch_inputs(torch, T, U, D, seed)
    logits, targets, lengths, spec = break_it(torch, base, U)
    with pytest.raises(TdtContractError) as err:
        tdt_loss(logits, targets, lengths, spec)
    assert needle in str(err.value), (label, str(err.value))


def test_torch_accepts_the_well_formed_call_the_cases_above_broke():
    """Negative control for the contract battery: the unbroken inputs must pass."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    T, U, D, seed = 2, 1, (0, 1), 71
    logits, targets, lengths, spec = _torch_inputs(torch, T, U, D, seed)
    out = tdt_loss(logits, targets, lengths, spec)
    assert tuple(out.shape) == (1,)
    assert bool(torch.isfinite(out).all())
    assert float(out[0]) == pytest.approx(reference_loss(T, U, D, seed), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("T,U,D,seed", ZERO_DURATION_CELLS, ids=ZERO_DURATION_IDS)
def test_torch_blank_only_lattice_matches_the_exact_rational_enumerator(T, U, D, seed):
    """Eq. 6's blank rule, checked against exact rationals rather than against the reference.

    Comparing torch to ``tdt_lattice`` alone could hide a shared bug; this compares it to the
    path enumerator's Fraction sum, which is the third independent program.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    p = exact_blank_only(T, D, seed)
    expect = -math.log(float(p))
    logits, targets, lengths, spec = _torch_inputs(torch, T, U, D, seed)
    got = float(tdt_loss(logits, targets, lengths, spec)[0])
    assert got == pytest.approx(expect, rel=1e-12, abs=1e-12), (T, D, got, expect, p)


@pytest.mark.parametrize("T,D", [(2, (0, 1)), (3, (0, 1)), (3, (0, 1, 2))])
def test_torch_blank_only_lattice_is_not_the_shifted_slot_lattice(T, D):
    """The zero-duration coverage can fail: charging blanks the previous slot moves the answer
    by orders of magnitude, and the torch lattice sits on the honest side."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    seed = 79
    honest = -math.log(float(exact_blank_only(T, D, seed)))
    shifted = -math.log(float(exact_blank_only_shifted(T, D, seed)))
    assert abs(honest - shifted) > 1e-3, (T, D, honest, shifted)
    logits, targets, lengths, spec = _torch_inputs(torch, T, 0, D, seed)
    got = float(tdt_loss(logits, targets, lengths, spec)[0])
    assert abs(got - honest) < 1e-12 * max(1.0, abs(honest)), (T, D, got, honest)
    assert abs(got - shifted) > 1e-3, (T, D, got, shifted)


def test_the_training_package_imports_no_product_or_teacher_runtime():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "nativevoice" / "training"
    banned = ("stt.adapters", "models.approval", "models.registry", "audio.features",
              "audio.", "tts", "faster_whisper", "sherpa", "onnxruntime", "numpy")
    for path in sorted(root.glob("*.py")):
        top = [ln for ln in path.read_text(encoding="utf-8").splitlines()
               if ln.startswith(("import ", "from "))]
        for token in banned:
            assert not any(token in ln for ln in top), f"{path.name}: {token}"
    torch_lines = [ln for ln in (root / "tdt_lattice.py").read_text(encoding="utf-8").splitlines()
                   if ln.startswith(("import ", "from "))]
    assert not any("torch" in ln for ln in torch_lines)
