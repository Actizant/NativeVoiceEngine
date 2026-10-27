"""Conformer-TDT student shape skeleton: real layers, random initialisation, no training.

This module builds an actual parameterised encoder - convolutional stem, input projection,
macaron feed-forwards, multi-head self-attention with an explicit relative-position variant, a
convolution module with a depthwise causal-style kernel, normalisation, a prediction network and
a joint TDT emission head that produces the slot layout ``nativevoice.training.tdt_loss``
consumes. Every layer is a real ``torch.nn`` module with real parameters.

What it is not: a trained model, a proposal for capacity, or evidence about speech. It has never
seen audio, no weights exist, and a finite loss from a random initialisation is arithmetic, not
recognition. The distinction is enforced by the tests, which assert shapes, finiteness and
gradient reachability and nothing else.

Two structural notes a reader will otherwise trip over.

* ``build_student`` defines its classes on each call rather than at module level, because
  ``tools/audit.py`` forbids a module-level heavy import anywhere under ``src/`` and a ``torch``
  class body needs ``torch.nn`` at definition time. The cost is a factory instead of an import;
  the benefit is that the project environment, which has no torch, still imports this package.
* Every architectural candidate is read from :class:`~nativevoice.modeling.config.StudentConfig`.
  None of them has a default here: attention positional variant, normalisation kind, conv-module
  activation, macaron ordering, predictor form, capacity, vocabulary, durations and the
  subsampling stack all arrive as explicit choices, and the open ones are listed in
  ``config.OPEN_ARCHITECTURE_DECISIONS``.

Padding, and what masking does and does not buy. ``encode`` and ``forward`` require
``frame_lengths`` - one explicit frame count per row - and derive every mask from it through
:mod:`nativevoice.modeling.masking`, never from a tensor dimension. Four masks are applied: the
padded frames of the input grid are zeroed, each subsampling stage zeroes the positions its row never
produced before the next stage reads them, attention may only attend to a row's own valid keys, and
the convolution module zeroes its activations between the first pointwise and the depthwise. Under
layer normalisation - the provisionally chosen direction, recorded in
``config.OPEN_ARCHITECTURE_DECISIONS`` - all four subsampling candidates reproduce the same row scored
alone at every valid position, to ``1.4e-15`` in float64, which is the measured floor of the arithmetic
and three orders inside the accepted tolerance. What would not be inert is ``batch_norm``, which stays
selectable: normalising over the length axis takes statistics from the pad columns and from the other
rows at every normalisation site in every block, so batch composition stays observable in training and
a padded batch writes running statistics that a clean row later reads. Selecting layer_norm is a
direction, not a default - ``normalization`` is still a required field.
"""

from __future__ import annotations

from nativevoice.core.errors import CapabilityUnavailable, ConfigurationError, VoiceError
from nativevoice.modeling.config import StudentConfig
from nativevoice.modeling.lengths import (
    UndefinedAxisError,
    frequency_bins,
    stage_length,
    stack_length,
)
from nativevoice.modeling.masking import position_lengths, valid_frame_mask, valid_position_mask


class ShapeContractError(VoiceError):
    """The built tensor does not have the length the shape arithmetic said it must have."""


def _torch():
    try:
        import torch
    except Exception as exc:                                       # env dependent
        raise CapabilityUnavailable(
            "the student shape skeleton needs a torch runtime; it is absent from the project "
            "environment and present only in the isolated verification environment",
            error=type(exc).__name__) from exc
    return torch


def shifted_targets(target_ids, *, start_symbol):
    """Prepend the start symbol so column ``u`` of the lattice sees ``u`` tokens.

    The start symbol is a required argument, not a constant: no vocabulary has been chosen in
    this repository, so there is no legitimate default for it.
    """
    torch = _torch()
    if target_ids.ndim != 2:
        raise ConfigurationError("target_ids must be [B, U]", ndim=target_ids.ndim)
    batch = target_ids.shape[0]
    head = torch.full((batch, 1), int(start_symbol), dtype=target_ids.dtype, device=target_ids.device)
    return torch.cat([head, target_ids], dim=1)


def derive_position_lengths(frame_lengths, *, batch, total_frames, variant):
    """Explicit per-row frame counts to per-row encoder position counts, via the named stack.

    This is the whole of D-A in one function: lengths are *stated* by the caller and then run
    through the same integer arithmetic the bridge uses, because a padded batch cannot say them
    to itself. A batch padded to the 30 s ceiling carries 748 positions for every row under the
    two-stage unpadded stack while the row holding one second of audio has 23, so reading
    ``states.shape[1]`` would overstate 725 of them.

    Refusals, all with the row index attached: a missing or empty sequence, a non-integer or a
    count above the padded frame axis, a count that cannot produce a position for this variant
    (``UndefinedAxisError`` from the stack arithmetic arrives as ``ShapeContractError``, naming the
    variant and the frame count), and a derived position count the padded axis cannot hold.
    """
    if isinstance(frame_lengths, (str, bytes)):
        raise ConfigurationError("frame_lengths must be a sequence of frame counts, not text",
                                 frame_lengths=frame_lengths)
    try:
        counts = tuple(frame_lengths)
    except TypeError as exc:
        raise ConfigurationError("frame_lengths must be a sequence of frame counts",
                                 frame_lengths=str(frame_lengths)) from exc
    if len(counts) != batch:
        raise ConfigurationError("frame_lengths must name every row of the batch exactly once",
                                 rows=len(counts), batch=batch)
    for index, frames in enumerate(counts):
        if isinstance(frames, bool) or not isinstance(frames, int):
            raise ConfigurationError("frame_lengths must be integer frame counts",
                                     row=index, frame_lengths=frames)
        if frames > total_frames:
            raise ConfigurationError("a declared frame length exceeds the padded frame axis",
                                     row=index, declared_frames=frames, padded_frames=total_frames)
    lengths = []
    for index, frames in enumerate(counts):
        try:
            lengths.append(position_lengths((frames,), variant=variant)[0])
        except UndefinedAxisError as exc:
            raise ShapeContractError(
                "a declared frame length is too short for the named subsampling stack",
                row=index, frames=frames, variant=variant.label,
                axis=exc.details.get("axis"), reason=str(exc)) from exc
    positions = tuple(lengths)
    for index, (frames, length) in enumerate(zip(counts, positions)):
        if length < 1:
            raise ConfigurationError(
                "a declared frame length yields no encoder position; a row of length zero is not "
                "a maskable row", row=index, frames=frames, variant=variant.label)
    padded_positions = stack_length(total_frames, variant=variant)
    if max(positions) > padded_positions:
        raise ConfigurationError("a declared length exceeds the padded position axis",
                                 declared_lengths=list(positions), padded_positions=padded_positions)
    return counts, positions, padded_positions


def build_student(config: StudentConfig):
    """Construct the skeleton described by ``config``. Raises without a torch runtime."""
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    d_model = config.d_model
    heads = config.attention_heads
    head_dim = d_model // heads
    dropout = float(config.dropout)
    macaron = config.macaron == "macaron"
    half = 0.5 if macaron else 1.0

    def make_norm(dim):
        if config.normalization == "layer_norm":
            return nn.LayerNorm(dim)
        return nn.BatchNorm1d(dim)

    def apply_norm(module, tensor):
        """LayerNorm works channel-last; BatchNorm1d wants channels in the middle."""
        if config.normalization == "layer_norm":
            return module(tensor)
        return module(tensor.transpose(1, 2)).transpose(1, 2)

    class Subsample(nn.Module):
        """One ``kernel x kernel`` stride-``stride`` convolution per stage, 1 channel in and out.

        Keeping the stem channel count at 1 is itself a choice: whether the stem projects into
        ``d_model`` channels before the input projection is an open architectural question, so it
        is not decided here. The geometry, however, is checked against the bridge on every call.
        """

        def __init__(self):
            super().__init__()
            spec = config.subsample
            self.spec = spec
            self.convs = nn.ModuleList(
                nn.Conv2d(1, 1, kernel_size=(spec.kernel, spec.kernel),
                          stride=(spec.stride, spec.stride), padding=(spec.padding, spec.padding))
                for _ in range(spec.stages))

        def forward(self, grid, frame_counts):
            """``frame_counts`` declares each row's valid frames; positions beyond it are zeroed.

            Each stage's own valid length is what the named stack makes of that declaration, and the
            positions past it are zeroed before the next stage can read them. This is arithmetic, not
            taste: scored alone, a row simply has no position at index ``A`` and the convolution
            substitutes a zero when the window reaches one; inside a padded batch that index exists
            and holds a real output - the bias, plus any of the row's own frames still inside the
            window - so zeroing it is the only way a *valid* position can mean the same number in
            both. Measured before the zeroing, the two-stage ``padding=1`` stack moved its last valid
            position by ``4.0e-1`` at 98 frames and by ``1.7e+0`` at 102. The axis lengths themselves
            are untouched, so the bridge, the seam check and every output formula still apply.
            """
            if grid.ndim != 3:
                raise ConfigurationError("the frontend grid must be [B, frames, mel]", shape=list(grid.shape))
            frames, mels = int(grid.shape[1]), int(grid.shape[2])
            if mels != config.mel_bins:
                raise ConfigurationError("the grid's mel axis disagrees with the config",
                                         grid_mels=mels, config_mels=config.mel_bins)
            try:
                want_frames = stack_length(frames, variant=self.spec)
                want_mels = frequency_bins(mels, variant=self.spec)
            except UndefinedAxisError as exc:
                raise ShapeContractError(
                    "this grid is too short for the named subsampling stack",
                    frames=frames, mels=mels, variant=self.spec.label, reason=str(exc)) from exc
            x = grid.unsqueeze(1)
            own = list(frame_counts)
            for conv in self.convs:
                x = conv(x)
                axis = int(x.shape[2])
                own = [stage_length(o, variant=self.spec) for o in own]
                keep = torch.tensor([[True] * min(o, axis) + [False] * (axis - min(o, axis))
                                     for o in own], dtype=torch.bool, device=grid.device)
                x = x.masked_fill(~keep.view(x.shape[0], 1, axis, 1), 0.0)
            if int(x.shape[2]) != want_frames or int(x.shape[3]) != want_mels:
                raise ShapeContractError(
                    "the convolution produced a length the shape arithmetic did not predict",
                    got_frames=int(x.shape[2]), want_frames=want_frames,
                    got_mels=int(x.shape[3]), want_mels=want_mels, variant=self.spec.label)
            return x.reshape(x.shape[0], x.shape[2], x.shape[3])

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            inner = config.feed_forward_mult * d_model
            self.a = nn.Linear(d_model, inner)
            self.b = nn.Linear(inner, d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.b(self.drop(functional.silu(self.a(x))))

    class Attention(nn.Module):
        """Scaled dot-product attention; positional information arrives only as a config choice."""

        def __init__(self):
            super().__init__()
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            self.drop = nn.Dropout(dropout)
            self.relative = None
            if config.position_encoding == "relative_shaw_windowed":
                self.relative = nn.Embedding(2 * config.relative_window + 1, head_dim)

        def forward(self, x, key_valid):
            """``key_valid`` is ``[B, T]``: True at the row's own positions. Padded keys never score.

            The mask is additive and applied before the softmax, so a valid query's weights are the
            softmax over valid keys only. Two properties the tests pin: the Shaw window still
            excludes what it always excluded (the masks intersect, they do not replace), and no row
            can end up with every entry masked, which is what a softmax of minus-infinity produces.
            A query past a row's end can fall outside the window of every valid key, so that row
            falls back to the pad mask alone - still at least one valid key, because a zero-length
            row is refused before the blocks run.
            """
            batch, length, _ = x.shape
            q = self.q(x).view(batch, length, heads, head_dim).transpose(1, 2)
            k = self.k(x).view(batch, length, heads, head_dim).transpose(1, 2)
            v = self.v(x).view(batch, length, heads, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(2, 3)) * (head_dim ** -0.5)
            keys = key_valid.view(batch, 1, 1, length)
            allowed = keys.expand(batch, heads, length, length)
            if self.relative is not None:
                window = config.relative_window
                index = torch.arange(length, device=x.device)
                delta = index.view(length, 1) - index.view(1, length)          # i - j
                gathered = torch.einsum("bchd,rd->bchr", q, self.relative.weight)
                pick = (delta + window).clamp(0, 2 * window).long()
                pick = pick.view(1, 1, length, length).expand(batch, heads, length, length)
                scores = scores + gathered.gather(3, pick)
                allowed = allowed & (delta.abs() <= window).view(1, 1, length, length)
                stranded = ~allowed.any(dim=-1, keepdim=True)
                allowed = torch.where(stranded, keys.expand_as(allowed), allowed)
            scores = scores.masked_fill(~allowed, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            merged = self.drop(weights @ v).transpose(1, 2).reshape(batch, length, d_model)
            return self.drop(self.out(merged))

    class ConvolutionModule(nn.Module):
        """Pointwise - depthwise - norm - activation - pointwise, with the depthwise kernel from config."""

        def __init__(self):
            super().__init__()
            inner = 2 * d_model if config.conv_activation == "glu" else d_model
            self.first = nn.Conv1d(d_model, inner, 1)
            self.depthwise = nn.Conv1d(inner, inner, kernel_size=config.conv_kernel_size,
                                       padding=(config.conv_kernel_size - 1) // 2, groups=inner)
            self.norm = make_norm(inner)
            # After the activation the channel count is d_model either way: glu halves the 2*d
            # inner width, swish leaves d alone. Writing this as `inner` is the bug that this
            # comment exists to prevent reintroducing.
            self.second = nn.Conv1d(d_model, d_model, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, pos_valid):
            """Channel-first inside, channel-last at the boundary, length preserved throughout.

            The zeroing sits between the first pointwise and the depthwise, and nowhere else does it
            work: the depthwise reaches ``(kernel-1)//2`` positions in each direction, so a pad
            column whose pointwise *bias* is nonzero is a nonzero input to a convolution that,
            scored alone, would have seen its own zero padding. Re-measured here on the fixture's
            ragged 98-into-160-frame batch with a kernel of 5, masking only the module input leaves
            the last valid position at ``1.7e-1``, masking only after the depthwise leaves
            ``3.4e-1``, and this placement leaves exactly ``0.0``.
            """
            length = int(x.shape[1])
            keep = pos_valid.view(x.shape[0], 1, length)
            h = self.depthwise(self.first(x.transpose(1, 2)).masked_fill(~keep, 0.0))
            if config.normalization == "batch_norm":
                h = self.norm(h)
            else:
                h = self.norm(h.transpose(1, 2)).transpose(1, 2)       # LayerNorm over channels
            h = functional.glu(h, dim=1) if config.conv_activation == "glu" else functional.silu(h)
            out = self.second(h).transpose(1, 2)                       # (B, T, d_model)
            if int(out.shape[1]) != length or int(out.shape[2]) != d_model:
                raise ShapeContractError(
                    "the convolution module changed the block's shape",
                    got=list(out.shape), want=[length, d_model],
                    activation=config.conv_activation, kernel=config.conv_kernel_size)
            return self.drop(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ff_left = FeedForward() if macaron else None
            self.norm_ff_left = make_norm(d_model) if macaron else None
            self.ff_right = FeedForward()
            self.norm_ff_right = make_norm(d_model) if macaron else None
            self.norm_final = None if macaron else make_norm(d_model)
            self.attn = Attention()
            self.norm_attn = make_norm(d_model)
            self.conv = ConvolutionModule()
            self.norm_conv = make_norm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, pos_valid):
            if self.ff_left is not None:
                x = x + half * self.ff_left(apply_norm(self.norm_ff_left, x))
            x = apply_norm(self.norm_attn, x + self.drop(self.attn(x, pos_valid)))
            x = apply_norm(self.norm_conv, x + self.drop(self.conv(x, pos_valid)))
            if self.norm_ff_right is not None:
                x = x + half * self.ff_right(apply_norm(self.norm_ff_right, x))
            else:
                x = apply_norm(self.norm_final, x + self.ff_right(x))
            return x

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(config.vocab_size, config.prediction_dim)
            self.lstm = None
            if config.prediction_kind == "lstm":
                self.lstm = nn.LSTM(config.prediction_dim, config.prediction_dim,
                                    num_layers=config.prediction_layers, batch_first=True)

        def forward(self, ids):
            hidden = self.embed(ids)
            if self.lstm is not None:
                hidden = self.lstm(hidden)[0]
            return hidden

    class EmissionHead(nn.Module):
        """Joint over every (position, prefix) pair, emitting the token slice then the duration slice."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(d_model, config.joint_dim)
            self.pred = nn.Linear(config.prediction_dim, config.joint_dim)
            self.token = nn.Linear(config.joint_dim, config.vocab_size)
            self.duration = nn.Linear(config.joint_dim, config.n_durations)

        def forward(self, states, prefixes):
            joined = torch.tanh(self.enc(states).unsqueeze(2) + self.pred(prefixes).unsqueeze(1))
            return torch.cat([self.token(joined), self.duration(joined)], dim=-1)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.subsample = Subsample()
            self.project = nn.Linear(frequency_bins(config.mel_bins, variant=config.subsample), d_model)
            self.blocks = nn.ModuleList(Block() for _ in range(config.block_count))
            self.predictor = Predictor()
            self.head = EmissionHead()
            self.parameters_total = sum(p.numel() for p in self.parameters())

        def encode(self, grid, *, frame_lengths):
            """States for a right-padded batch, with every mask derived from ``frame_lengths``.

            ``frame_lengths`` is required and keyword-only. It states, per row, how many frames of
            the grid are audio; the position count for that row is what the named stack makes of it,
            never what ``grid.shape[1]`` suggests. The returned axis is the padded one.
            """
            torch = _torch()
            if grid.ndim != 3:
                raise ConfigurationError("the frontend grid must be [B, frames, mel]",
                                         shape=list(grid.shape))
            batch, total_frames = int(grid.shape[0]), int(grid.shape[1])
            counts, lengths, total_positions = derive_position_lengths(
                frame_lengths, batch=batch, total_frames=total_frames, variant=config.subsample)
            frames_on = torch.tensor(valid_frame_mask(total_frames, counts),
                                     dtype=torch.bool, device=grid.device)
            positions_on = torch.tensor(valid_position_mask(total_positions, lengths),
                                        dtype=torch.bool, device=grid.device)
            x = grid.masked_fill(~frames_on.unsqueeze(-1), 0.0)
            x = self.subsample(x, counts)
            if int(x.shape[1]) != total_positions:
                raise ShapeContractError("the stem produced an axis the derived lengths do not predict",
                                         got=int(x.shape[1]), want=total_positions,
                                         variant=config.subsample.label)
            x = self.project(x)
            for block in self.blocks:
                x = block(x, positions_on)
            return x

        def forward(self, grid, target_ids, *, frame_lengths):
            """Returns ``(logits, per_row_positions, padded_positions)``.

            ``logits`` is ``[B, positions, U+1, vocab + Nd]``. The second value is each row's own
            ``T_b`` - the length its lattice must be told - derived from ``frame_lengths`` through
            :func:`derive_position_lengths`, the same single authority the masks use. The third is
            the padded axis the tensor was built to. They differ for every batch that is not
            square, and a single number covering both is what made the wrong one easy to pass:
            handing the loss the padded axis for a short row scores positions the utterance never
            produced, and the extra rows are inert, so the loss stays finite. The two are therefore
            returned apart and neither is privileged.
            """
            states = self.encode(grid, frame_lengths=frame_lengths)
            _, per_row_positions, padded_positions = derive_position_lengths(
                frame_lengths, batch=int(grid.shape[0]), total_frames=int(grid.shape[1]),
                variant=config.subsample)
            if target_ids.ndim != 2 or int(target_ids.shape[0]) != int(grid.shape[0]):
                raise ConfigurationError(
                    "target_ids must carry one target sequence per row of the grid: a batch-size-1 "
                    "target tensor broadcasts silently across the whole batch, so every row would "
                    "be scored against the same transcript",
                    grid_rows=int(grid.shape[0]),
                    target_shape=list(target_ids.shape) if hasattr(target_ids, "shape") else None,
                    ndim=int(target_ids.ndim))
            prefixes = self.predictor(shifted_targets(target_ids,
                                                     start_symbol=config.prediction_start_symbol))
            logits = self.head(states, prefixes)
            if int(logits.shape[1]) != int(states.shape[1]):
                raise ShapeContractError("emission axis disagrees with encoder positions",
                                         got=int(logits.shape[1]), want=int(states.shape[1]))
            if padded_positions != int(states.shape[1]):
                raise ShapeContractError("the derived padded axis disagrees with the states tensor",
                                         derived=padded_positions, actual=int(states.shape[1]))
            return logits, per_row_positions, int(states.shape[1])

    return Student()
