"""Reproducible mathematical verification of the TDT lattice and decoder contract.

Two independently authored implementations are compared on every lattice:
``tdt_lattice_oracle`` enumerates emission sequences; ``tdt_lattice_recursion`` runs a
recursion. Neither calls the other. Mutation flags exist so the suite can prove it would fail on a wrong
implementation.

Run from this directory with the repository on PYTHONPATH and bytecode disabled:

    python -m pytest tests/test_tdt_lattice.py -q

Nothing here trains a model, touches the repository, or measures accuracy.
"""

from fractions import Fraction

import pytest

from tdt_lattice_oracle import BLANK, V1, V2, V3, completion_probability, enumerate_paths, is_admissible, random_tables, total_probability
from tdt_lattice_recursion import (Status, cut_total, decode, forward_alpha, p_tdt, backward_alpha_free)

LATTICES = [(T, U, D, seed)
            for T in (1, 2, 3)
            for U in (0, 1, 2, 3)
            for D in ((0, 1), (1,), (0, 1, 2))
            for seed in (T * 100 + U * 10 + len(D),)]
LATTICE_IDS = [f"T{T}-U{U}-D{''.join(map(str, D))}" for T, U, D, _ in LATTICES]

CONVENTIONS = [(V3, "V3"), (V1, "V1"), (V2, "V2")]


def tables(T, U, D, seed):
    return random_tables(T, U, D, seed)


# ------------------------------------------------------------------ 1. recursion vs enumeration


@pytest.mark.parametrize("T,U,D,seed", LATTICES, ids=LATTICE_IDS)
@pytest.mark.parametrize("conv,label", CONVENTIONS, ids=["V3", "V1", "V2"])
def test_forward_recursion_matches_exhaustive_enumeration(T, U, D, seed, conv, label):
    Pt, Pd = tables(T, U, D, seed)
    a, terminal = forward_alpha(T, U, Pt, Pd, D, conv)
    recursive = a.get(terminal, Fraction(0))
    enumerated = total_probability(T, U, Pt, Pd, D, conv)
    assert recursive == enumerated, (label, T, U, D)
    # The old form of this line was `len(...) >= 1 or enumerated == 0`, which cannot fail:
    # `total_probability` is the sum over the very enumeration being counted, so an empty
    # enumeration makes the right-hand disjunct true by construction. What is actually worth
    # asserting is the biconditional - no paths exactly when the total is zero - plus that every
    # path the enumerator produced carries a strictly positive weight.
    paths = list(enumerate_paths(T, U, Pt, Pd, D, conv))
    assert (len(paths) == 0) == (enumerated == 0), (label, T, U, D)
    assert all(weight > 0 for weight, _ in paths), (label, T, U, D)


def test_v1_and_v3_agree_in_value_and_v2_differs():
    """The dead-end variant cannot change the loss; the sentinel-state variant can."""
    same_v3_v1 = True
    larger_v2 = False
    for T, U, D, seed in LATTICES:
        if U == 0:
            continue
        Pt, Pd = tables(T, U, D, seed)
        p3 = p_tdt(T, U, Pt, Pd, D, V3)
        p1 = p_tdt(T, U, Pt, Pd, D, V1)
        p2 = p_tdt(T, U, Pt, Pd, D, V2)
        same_v3_v1 &= (p3 == p1)
        n3 = len(enumerate_paths(T, U, Pt, Pd, D, V3))
        n2 = len(enumerate_paths(T, U, Pt, Pd, D, V2))
        if n2 > n3:
            larger_v2 = True
    assert same_v3_v1
    assert larger_v2


# ------------------------------------------------------------------ 2. forward/backward cut identity


@pytest.mark.parametrize("T,U,D,seed", LATTICES[:24], ids=LATTICE_IDS[:24])
def test_cut_identity_holds_on_every_diagonal(T, U, D, seed):
    Pt, Pd = tables(T, U, D, seed)
    total = total_probability(T, U, Pt, Pd, D, V3)
    for n in range(2, (T + 1) + U + 1):
        assert cut_total(T, U, Pt, Pd, D, n, V3) == total, (n, T, U, D)


@pytest.mark.parametrize("T,U,D,seed", LATTICES[:24], ids=LATTICE_IDS[:24])
def test_backward_recursion_matches_enumerrated_completion(T, U, D, seed):
    Pt, Pd = tables(T, U, D, seed)
    b_rec = backward_alpha_free(T, U, Pt, Pd, D, V3)
    for t in range(1, T + 2):
        for u in range(U + 1):
            assert b_rec.get((t, u), Fraction(0)) == completion_probability(T, U, Pt, Pd, D, (t, u), V3), (t, u)


# ------------------------------------------------------------------ 3. terminal boundary conditions


def test_alpha_base_condition_and_terminal_node_value():
    T, U, D = 3, 2, (0, 1, 2)
    Pt, Pd = tables(T, U, D, 7)
    a, _ = forward_alpha(T, U, Pt, Pd, D, V3)
    assert a[(1, 0)] == Fraction(1)
    assert completion_probability(T, U, Pt, Pd, D, (T + 1, U), V3) == Fraction(1)


def test_v3_zeroes_the_terminal_column_except_the_target_complete_state():
    T, U, D = 3, 2, (0, 1, 2)
    Pt, Pd = tables(T, U, D, 7)
    a, _ = forward_alpha(T, U, Pt, Pd, D, V3)
    b = backward_alpha_free(T, U, Pt, Pd, D, V3)
    for u in range(U):
        assert a.get((T + 1, u), Fraction(0)) == 0
        assert b.get((T + 1, u), Fraction(0)) == 0


def test_v2_violates_the_printed_boundary_sentence():
    """The source states beta(T+1,u)=0 for all u != U; the sentinel-state variant breaks it."""
    T, U, D = 3, 2, (0, 1, 2)
    Pt, Pd = tables(T, U, D, 7)
    assert completion_probability(T, U, Pt, Pd, D, (T + 1, U - 1), V2) > 0


def test_beta_at_T_equals_the_single_closing_blank():
    """beta(T,U) is not assumed; it is derived as P(blank, duration 1 | T, U)."""
    for T, U, D, seed in LATTICES:
        if 1 not in D or U == 0:
            continue
        Pt, Pd = tables(T, U, D, seed)
        b = backward_alpha_free(T, U, Pt, Pd, D, V3)
        derived = Fraction(Pt[(T, U)][BLANK]) * Fraction(Pd[(T, U)][1])
        assert b[(T, U)] == derived, (T, U, D)
        # the marginal reading (any positive duration closing the lattice) is refuted
        if len([d for d in D if d > 1]):
            marginal = sum(Fraction(Pt[(T, U)][BLANK]) * Fraction(Pd[(T, U)][d])
                           for d in D if d > 0)
            assert marginal != derived, (T, U, D)


# ------------------------------------------------------------------ 4. duration sum and admissibility


@pytest.mark.parametrize("T,U,D,seed", LATTICES[:24], ids=LATTICE_IDS[:24])
def test_every_enumerated_path_sums_durations_to_T(T, U, D, seed):
    Pt, Pd = tables(T, U, D, seed)
    for _, emissions in enumerate_paths(T, U, Pt, Pd, D, V3):
        assert sum(d for _, d in emissions) == T
        assert all(d > 0 for v, d in emissions if v == BLANK)
        assert is_admissible(T, U, D, emissions, V3)


def test_admissibility_checker_rejects_a_synthetic_overshoot():
    T, U, D = 2, 1, (0, 1, 2)
    assert not is_admissible(T, U, D, ((f"y1", 2), (BLANK, 2)))       # sums to 4 != T
    assert not is_admissible(T, U, D, ((BLANK, 0), (f"y1", 1)))       # blank duration 0
    assert is_admissible(T, U, D, ((f"y1", 1), (BLANK, 1)))           # exactly T


# ------------------------------------------------------------------ 5. zero-duration token necessity


@pytest.mark.parametrize("T,U", [(1, 2), (1, 3), (2, 3), (3, 5)])
def test_likelihood_is_exactly_zero_without_zero_duration_tokens_when_U_exceeds_T(T, U):
    Pt, Pd = tables(T, U, (1,), 11)
    assert total_probability(T, U, Pt, Pd, (1,), V3) == 0
    assert p_tdt(T, U, Pt, Pd, (1,), V3) == 0
    Pt2, Pd2 = tables(T, U, (0, 1), 11)
    assert total_probability(T, U, Pt2, Pd2, (0, 1), V3) > 0


Z_LATTICES = [(T, U, D, T * 100 + U * 10 + len(D))
              for T in (1, 2, 3, 4)
              for U in (1, 2, 3, 4)
              for D in ((0, 1), (0, 1, 2))]
Z_IDS = [f"T{T}-U{U}-D{''.join(map(str, D))}" for T, U, D, _ in Z_LATTICES]


@pytest.mark.parametrize("T,U,D,seed", Z_LATTICES, ids=Z_IDS)
def test_zero_duration_tokens_are_exactly_the_excess_over_the_frame_count(T, U, D, seed):
    """Removing zero-duration tokens can only lose probability, and kills the likelihood
    precisely when the target is longer than the encoder output. No skipped cases."""
    Pt, Pd = tables(T, U, D, seed)
    honest = p_tdt(T, U, Pt, Pd, D, V3)
    starved = p_tdt(T, U, Pt, Pd, D, V3, token_requires_positive=True)
    assert starved <= honest, (T, U, D)
    if U > T:
        assert starved == 0, (T, U, D, float(honest))
    else:
        assert starved > 0, (T, U, D, float(honest))


# ------------------------------------------------------------------ 6. decoder classification


def fixed(T, U, D, token_pref, duration_pref):
    """Deterministic tables: which label wins, and which duration wins.

    ``token_pref`` is ``"NONBLANK"`` to prefer any real token over the blank, or ``BLANK``
    to prefer the blank.
    """
    Pt, Pd = {}, {}
    names = [BLANK] + [f"y{i+1}" for i in range(U)]
    for t in range(1, T + 2):
        for u in range(U + 1):
            if token_pref == BLANK:
                Pt[(t, u)] = {n: (0.9 if n == BLANK else 0.01) for n in names}
            else:
                Pt[(t, u)] = {n: (0.01 if n == BLANK else 0.9) for n in names}
            Pd[(t, u)] = {d: (0.9 if d == duration_pref else 0.01) for d in D}
    return Pt, Pd


CASES = [
    ("zero-duration tokens are legitimate", 2, 3, (0, 1), "NONBLANK", 0, Status.COMPLETE, 3),
    ("blank with duration 0 must advance", 3, 2, (0, 1), BLANK, 0, Status.END_OF_INPUT_UNCONSUMED, 0),
    ("token at the output limit must advance", 2, 2, (0, 1), "NONBLANK", 0, Status.COMPLETE, 2),
    ("invalid duration configuration", 3, 1, (0,), "NONBLANK", 0, Status.NO_POSITIVE_DURATION_AVAILABLE, 0),
    ("overshoot is a zero-likelihood alignment", 3, 1, (0, 2), "NONBLANK", 2, Status.OVERSHOOT_ZERO_LIKELIHOOD, 0),
]


@pytest.mark.parametrize("name,T,U,D,tp,dp,expected,expected_zero",
                         CASES, ids=[c[0] for c in CASES])
def test_decoder_reports_the_named_outcome(name, T, U, D, tp, dp, expected, expected_zero):
    Pt, Pd = fixed(T, U, D, tp, dp)
    out = decode(T, U, Pt, Pd, D)
    assert out.status is expected, (name, out)
    assert out.zero_duration_tokens == expected_zero, (name, out)
    if out.status is Status.COMPLETE:
        assert is_admissible(T, U, D, out.trace, V3), (name, out)


def test_plausible_transcript_from_an_invalid_path_is_not_accepted():
    """Case 6: a full-length hypothesis whose durations overshoot the terminal node."""
    T, U, D = 3, 1, (0, 2)
    Pt, Pd = fixed(T, U, D, "NONBLANK", 2)
    broken = decode(T, U, Pt, Pd, D, terminal_validation=False)
    assert broken.status is Status.COMPLETE and len(broken.hypothesis) == U
    assert not is_admissible(T, U, D, broken.trace, V3)
    good = decode(T, U, Pt, Pd, D)
    assert good.status is Status.OVERSHOOT_ZERO_LIKELIHOOD
    with pytest.raises(Exception):
        good.transcript()


def test_only_complete_may_be_transcribed():
    for T, U, D, tp, dp in [(4, 3, (0, 1), BLANK, 1), (3, 1, (0,), "y1", 0)]:
        Pt, Pd = fixed(T, U, D, tp, dp)
        out = decode(T, U, Pt, Pd, D)
        assert out.status is not Status.COMPLETE
        with pytest.raises(Exception):
            out.transcript()


def test_inference_policies_are_explicit_and_tested():
    T, U, D = 3, 4, (0, 1)
    Pt, Pd = fixed(T, U, D, "NONBLANK", 0)
    cap = 2
    capped = decode(T, U, Pt, Pd, D, max_symbols_per_step=cap)
    assert capped.status is Status.COMPLETE and capped.cap_fired > 0
    # the cap is a per-audio-frame limit on appends: verify it directly from the positions
    positions = []
    t = 1
    for token, duration, appended in capped.trace:
        if appended:
            positions.append(t)
        t += duration
    counts = {p: positions.count(p) for p in set(positions)}
    assert max(counts.values()) <= cap, counts
    assert capped.zero_duration_tokens > 0, "the cap must not erase legitimate zero-duration tokens"

    limited = decode(T, U, Pt, Pd, D, output_length_limit=2)
    assert limited.status is Status.OUTPUT_LENGTH_LIMIT_REACHED
    with pytest.raises(Exception):
        limited.transcript()
    stalled = decode(T, U, Pt, Pd, D, progress_obligation=False, max_iterations=5)
    assert stalled.status is Status.STALL_MAX_ITERATIONS


def test_non_appending_steps_always_advance():
    """The progress obligation, checked step by step rather than inferred."""
    for seed in range(200):
        T = 1 + seed % 5
        U = 1 + (seed // 5) % 4
        D = (0, 1, 2)
        Pt, Pd = tables(T, U, D, seed)
        out = decode(T, U, Pt, Pd, D)
        for token, duration, appended in out.trace:
            if not appended:
                assert duration > 0, (seed, token, duration, out)
            if appended:
                assert token != BLANK and duration in D, (seed, out)
        assert sum(d for _, d, _ in out.trace) == out.position - 1, (seed, out)


# ------------------------------------------------------------------ 7. progress and capability preservation


def test_progress_obligation_removes_every_stall():
    """Without the obligation the hazard is real; with it, no lattice in the sweep stalls."""
    naked_stalls = guarded_stalls = 0
    inadmissible = 0
    completions = 0
    for seed in range(400):
        T = 1 + seed % 6
        U = 1 + (seed // 6) % 5
        D = (0, 1, 2)
        Pt, Pd = tables(T, U, D, seed)
        naked = decode(T, U, Pt, Pd, D, progress_obligation=False, terminal_validation=False,
                       max_iterations=200)
        guarded = decode(T, U, Pt, Pd, D, max_iterations=200)
        naked_stalls += naked.status is Status.STALL_MAX_ITERATIONS
        guarded_stalls += guarded.status is Status.STALL_MAX_ITERATIONS
        if guarded.status is Status.COMPLETE:
            completions += 1
            if not is_admissible(T, U, D, guarded.trace, V3):
                inadmissible += 1
    assert naked_stalls > 0, "the decoder hazard was not reproduced; the test would be vacuous"
    assert guarded_stalls == 0
    assert completions > 0
    assert inadmissible == 0


def test_progress_obligation_does_not_remove_legitimate_zero_duration_tokens():
    """On every trajectory that already terminated, the guard must change nothing."""
    compared = 0
    for seed in range(400):
        T = 1 + seed % 6
        U = 1 + (seed // 6) % 5
        D = (0, 1, 2)
        Pt, Pd = tables(T, U, D, seed)
        naked = decode(T, U, Pt, Pd, D, progress_obligation=False, max_iterations=200)
        guarded = decode(T, U, Pt, Pd, D, max_iterations=200)
        if naked.status is not Status.STALL_MAX_ITERATIONS:
            assert naked.zero_duration_tokens == guarded.zero_duration_tokens, seed
            assert naked.hypothesis == guarded.hypothesis, seed
            compared += 1
    assert compared > 50, f"only {compared} runs were comparable; the check is too weak"


# ------------------------------------------------------------------ 8. mutation controls


def test_mutations_are_detected_by_the_enumeration_oracle():
    """A test suite that cannot fail is not evidence. Each rule is asserted on its own, never in an `or`.

    Three limbs, three different expectations, measured on this fixture: ``terminal_column_T`` and
    ``token_requires_positive`` both change the exact probability and must be caught.
    ``allow_blank_zero`` must *not* change it - a zero-duration blank is a self-loop on the same
    lattice state, and the forward sweep correctly refuses to add it - so the equality is the
    assertion that keeps the rule honest. Without this split the blank-zero limb could never fail and
    the disjunction would still pass on the strength of the other two.
    """
    for T, U, D, seed in [(3, 2, (0, 1), 7), (2, 3, (0, 1), 5), (3, 3, (0, 1, 2), 9)]:
        Pt, Pd = tables(T, U, D, seed)
        honest = p_tdt(T, U, Pt, Pd, D, V3)
        assert honest > 0, (T, U, D)
        assert p_tdt(T, U, Pt, Pd, D, V3, terminal_column_T=True) != honest, (T, U, D, "terminal rule")
        assert p_tdt(T, U, Pt, Pd, D, V3, token_requires_positive=True) != honest, (
            T, U, D, "zero-duration token rule")
        assert p_tdt(T, U, Pt, Pd, D, V3, allow_blank_zero=True) == honest, (
            T, U, D, "a self-loop blank must not add mass")


def test_oracle_and_recursion_are_not_the_same_program():
    """Guard against the suite degenerating into a tautology."""
    import inspect

    from tdt_lattice_oracle import total_probability as oracle_total
    src_oracle = inspect.getsource(oracle_total)
    src_spec = inspect.getsource(forward_alpha)
    assert "walk(" in src_oracle or "enumerate_paths" in src_oracle
    assert "for order in" in src_spec
    assert "walk(" not in src_spec


# ------------------------------------------------------------------ 9. length adapter arithmetic


def test_frontend_frame_law_matches_the_frozen_code():
    from nativevoice.audio.features import FBankConfig
    cfg = FBankConfig()
    assert (cfg.window_samples, cfg.hop_samples, cfg.center, cfg.tail_pad_samples,
            cfg.drop_last_frame) == (400, 160, "none", 0, False)
    for N in list(range(0, 6000)) + list(range(6000, 60001, 17)):
        expected = 0 if N < 400 else 1 + (N - 400) // 160
        assert cfg.frame_count(N) == expected, N


def test_convolution_length_two_independent_derivations_agree():
    def composed(T):
        def conv(x, k=3, s=2):
            return 0 if x < k else (x - k) // s + 1
        return conv(conv(T))

    def closed_form(T, rf=7, stride=4):          # RF = 3 + 2*(3-1), stride = 2*2
        return 0 if T < rf else (T - rf) // stride + 1

    assert [composed(T) for T in range(6001)] == [closed_form(T) for T in range(6001)]
    # stride-8 stack cross-check: RF = 3 + 2*2 + 4*2 = 15
    def conv3(T):
        def conv(x, k=3, s=2):
            return 0 if x < k else (x - k) // s + 1
        return conv(conv(conv(T)))
    assert [conv3(T) for T in range(6001)] == [closed_form(T, 15, 8) for T in range(6001)]


def test_boundary_and_cadence_values():
    from nativevoice.audio.features import FBankConfig
    cfg = FBankConfig()

    def conv(x):
        return 0 if x < 3 else (x - 3) // 2 + 1
    def Tprime(N):
        return conv(conv(cfg.frame_count(N)))

    assert Tprime(1359) == 0 and Tprime(1360) == 1
    assert min(N for N in range(1, 60000) if Tprime(N) >= 1) == 1360
    assert cfg.frame_count(16000) == 98 and Tprime(16000) == 23
    assert cfg.frame_count(480000) == 2998 and Tprime(480000) == 748
    assert FBankConfig.whisper_parity().frame_count(16000) == 101


# ------------------------------------------------- adapter-to-lattice seam (assumption under test)


def test_adapter_length_and_lattice_length_are_the_same_number():
    """ASSUMPTION, not an existing product component: the stride-4 stack is two unpadded
    3x3 stride-2 convolutions applied to the frozen frontend grid. No encoder exists in this
    repository, so this test binds the seam that will hand a length to the lattice.

    A one-frame error here is silent in training: the lattice would require durations to sum
    to a T the encoder never produced. So the test pins the boundary pair, shows every
    intermediate length, and proves the lattice *notices* a wrong T.
    """
    from nativevoice.audio.features import FBankConfig
    cfg = FBankConfig()

    def conv(x, k=3, s=2):
        return 0 if x < k else (x - k) // s + 1

    def lattice_T(N):
        return conv(conv(cfg.frame_count(N)))

    # boundary pair, with the frontend's own intermediate lengths shown
    assert cfg.frame_count(1359) == 6 and lattice_T(1359) == 0      # no frames, no emissions
    assert cfg.frame_count(1360) == 7 and conv(7) == 3 and lattice_T(1360) == 1
    assert cfg.frame_count(3600) == 21 and conv(21) == 10 and conv(conv(21)) == 4
    assert lattice_T(16000) == 23 and lattice_T(48000) == 73 and lattice_T(480000) == 748

    # the seam itself: the lattice must be handed exactly this number
    T = lattice_T(3600)                                             # == 4, small enough to enumerate
    U, D = 3, (0, 1, 2)
    Pt, Pd = random_tables(T + 1, U, D, 3)                          # tables cover both lengths
    correct = total_probability(T, U, Pt, Pd, D, V3)
    off_by_one = total_probability(T + 1, U, Pt, Pd, D, V3)
    assert correct > 0 and off_by_one > 0
    assert correct != off_by_one, "a one-frame length error was invisible to the lattice"
    assert p_tdt(T, U, Pt, Pd, D, V3) == correct                    # recursion agrees at the true T
    assert p_tdt(T + 1, U, Pt, Pd, D, V3) == off_by_one

    # and the duration-sum law follows whichever length is declared, so a mismatch is
    # detectable from the trace alone
    for declared in (T, T + 1):
        paths = enumerate_paths(declared, U, Pt, Pd, D, V3)
        assert paths
        for _, emissions in paths:
            assert sum(d for _, d in emissions) == declared
