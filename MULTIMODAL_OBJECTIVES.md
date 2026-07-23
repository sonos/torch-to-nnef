# Multimodal joint-export: current objectives

Two objectives for the multimodal encoder work on this branch. Objective 1
completes the last Gemma 4 tower on the current (baked-input-size) design;
Objective 2 removes the baked-size limitation across all towers.

## Objective 1 — Gemma 4 audio tower to tract

Finish the third Gemma 4 branch so audio joins vision + video in the joint
tract export.

- **State today:** the recipe is validated in PyTorch
  (`test_dummy_audio_chain_parity`, exact match); `Gemma4AudioEncoder` /
  `Gemma4AudioEncoderHandler` exist but are **not registered**.
- **The only tract blocker:** `masked_fill` lowers to NNEF `select`, and
  tract's `select` requires a **bool** condition
  (`inputs[0].datum_type.is::<bool>()`). The audio attention masks with a
  baked **float** mask, so tract rejects it. Everything else in the conformer
  (subsample conv2d, `unfold`, chunked 5-D local attention, `_rel_shift`,
  causal conv1d) already exports — confirmed by running the export.
- **Two ways to fix (pick one):**
  - **(A) handler-side additive mask** — replace
    `masked_fill(mask.logical_not(), -1e9)` with adding a constant `0 / -1e9`
    float bias before softmax. Numerically identical, no bool `select`, no core
    change. Fits the baked-chunk design.
  - **(B) general t2n fix** — make the `masked_fill` / `where` → `select`
    lowering emit the condition as a genuine tract bool. Broader benefit (any
    masked_fill model), touches shared lowering.
- **Done when:** audio encoder registered + `"audio"` added to the Gemma 4
  decoder `MODALITIES`; dummy export + tract `check_io` green at a baked chunk
  length (consistent with the Voxtral audio tower). Chain-parity already covers
  correctness.
- **Caveat:** the export currently fails at the *first* `select`; fixing the
  mask may reveal a further blocker or a numeric `check_io` gap behind it.

## Objective 2 — v2: dynamic-size encoders

Un-bake encoder input size so **one** exported graph handles variable
resolution / audio duration, instead of one graph per fixed size. Removes the
single-resolution (Qwen2.5-VL, Gemma 4 vision) and single-chunk (Voxtral,
Gemma 4 audio) limitation of the baked v1 towers.

- **Approach:** per tower, make the patch/frame axis symbolic
  (`dynamic_axes`), export, fix the first symbolic-shape blocker (t2n-side, or
  flag if genuinely tract-side), repeat; validate with tract at ≥2 concrete
  sizes.
- **First target:** Gemma 4 vision — its position embedding is a table gather
  by 2D position id (no interpolation / `grid_sample`, which tract lacks) and
  it uses plain full attention (no window / chunk complications).
- **Blocker #1 (found, t2n-side, core):** shape queries on **static** axes do
  not constant-fold once any axis of the tensor is dynamic. In the 2-D RoPE,
  `x.shape[-1]` (head_dim) and `position_ids.shape[-1]` (=2) resolve to `None`
  under a dynamic patch axis, so `split_with_sizes` fails on `int(None)`. Fix =
  per-axis static/dynamic resolution in shape / `aten::size` handling. Benefits
  every dynamic export. **This is the next task; it touches a widely-used core
  path, so land it with the full core + LLM regression suites run before/after.**
- **Expected later blockers:** symbolic-`S` attention reshapes; the Gemma 4
  vision **pooler** (2-D avg-pool groups patches into
  `output_length = P // pool**2` via `one_hot(kernel_idxs, symbolic)` — the hard
  one); then per-tower specials (Qwen `window_index` / `cu_seqlens` from
  `grid_thw`; audio `num_blocks = ceil(S / chunk)` + `unfold` + `pad`).
- **Relationship to Objective 1:** v2 eventually makes audio dynamic too, so it
  subsumes O1's baking. O1 still ships usable audio now; O2 is the longer,
  multi-step core-t2n effort.
