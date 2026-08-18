# Refreshing the measured ONNX support data

The `ONNX` section of [supported operators](./supported_operators.md) is
**measured**, not scraped. This page explains how it is produced, how to
refresh it, and how to read a grade.

## Why it is measured

PyTorch used to publish `onnx_torchscript_supported_aten_ops.html`, a
per-operator table that this page scraped. That table described the
TorchScript exporter, which was **removed in torch 2.9**, and the page
404s from that version on. The generator kept working only by falling back
to the torch 2.8 copy, so the column described an exporter that no longer
exists and could not be refreshed.

The `dynamo` exporter that replaced it does not ship an equivalent table.
But the [proptest catalog](./internal_design.md) already builds real
modules covering 440+ `aten::` operators, so the support level can simply
be measured: export each one and record what happens.

Measuring also surfaces something a doc table cannot express. Support is
frequently **partial**: an operator exports at f32 rank-2 and raises at
f16 or rank 0. Sweeping shapes and dtypes finds that; a flag cannot.

## Refreshing

```bash
tox -e proptest_onnx                         # measure, writes the artifact
python docs/contributing/generate_support_page.py \
    --onnx-report docs/contributing/onnx_support_measured.json
```

The first command writes
`docs/contributing/onnx_support_measured.json`; the second renders it into
the page. Both the artifact and the page are committed, so a grade change
is visible in review as a diff instead of as a difference between two runs
nobody can compare.

Without `--onnx-report`, the generator falls back to scraping the retired
torch 2.8 listing, exactly as before.

`--torch-version` defaults to the installed torch, which is the same one
the sweep measured, so the operator list and the measured column cannot
silently describe different releases. Pass it explicitly only to target a
version other than the one installed.

**Bumping torch** therefore means editing one place: the `torch==` pin in
`[tool.tox.env.proptest_onnx]`. The doc-link cache for the new version is
built automatically, and every grade is re-measured because the version is
part of the reuse fingerprint.

**Adding operator coverage** needs no changes here at all: the sweep is
parametrized over the spec registry, so a new spec with `aten_ops` is
measured on the next run and appears on the next regeneration.

### Cost and reuse

A full sweep is ~510 specs x 25 examples, each one an export plus an
onnxruntime run. To keep repeat regenerations cheap, a grade of `full` can
be **carried over** from the previous artifact, but only when:

- every operator the spec declares was graded `full`, and
- the recorded environment fingerprint (torch, onnx, onnxruntime,
  onnxscript, opset, exporter path) matches the current one exactly, and
- that measurement drew at least as many examples as the current profile.

Anything else is re-measured. `partial` and `none` are **never** reused:
`none -> full` is the common transition as PyTorch adds support, so
re-measuring those every time is where the value is.

Reuse is not permanent. Each regeneration also force-re-measures a
rotating tenth of the operators (`regen_index`), so no `full` grade can be
carried forever without being checked. `--onnx-no-reuse` forces a complete
sweep.

The fingerprint check is what makes reuse safe rather than optimistic. The
2.8 -> 2.9 exporter swap regressed every supported operator at once; only
a version comparison catches that class of change.

### Profiles

The tox env pins the `ci` hypothesis profile, which derandomizes: the
committed artifact is then reproducible, and a diff means a real change
rather than different draws. For a deeper sweep:

```bash
T2N_HYP_PROFILE=nightly tox -e proptest_onnx   # 200 examples, several hours
```

A `nightly` result supersedes a `ci` one for reuse purposes (more examples
is strictly more evidence); the reverse is refused.

## Reading the headline bars

Both bars use the **same denominators as the `TractNNEF` tab**: the core
opset size first, then the full `aten::` listing. That is what makes the
two tabs comparable at a glance; a bar over "the operators we measured"
renders near-full whatever the coverage is, and sits next to a bar that
means something else entirely.

The numerator is the number of rows we measured as `full`. If a future
torch listing adds a targetable row, add a spec for it before refreshing
the page. If the row cannot be tied to an attributable proptest graph,
record it in `tests/proptest/op_specs/untranslated.py` so the generator
keeps it outside both denominators and explains why in the appendix.

The caption under the bars keeps the strict "of what we actually tested,
how much passed" ratio one line away.

## Reading a grade

The `export` column grades operator coverage only:

| Glyph | Meaning |
| --- | --- |
| ✅ `full` | every generated example exported |
| 🟡 `partial` | some examples exported, others raised. The artifact keeps the failing shapes/dtypes |
| ❌ `none` | no example exported, and the exporter refused at least one |
| ⚠️ `blocked` | `torch.export` could not capture the module, so the ONNX exporter never ran. **Not** an ONNX verdict |
| `-` | no spec covers it, so we did **not** measure it here. **Not** unsupported |

Only ✅ / 🟡 / ❌ are measurements. `-` means unmeasured, not failed.
A `-` row should be temporary: add a direct spec when the row is
targetable, or move it to `untranslated.EXCLUDED` when it is not. The
`documented` column records what the retired listing claimed, and the
`spec coverage` column says whether this row was measured or why it has
no direct proptest measurement. Rows that cannot be attributed to a
proptest graph target are filtered out of both comparison tables and
listed in the appendix instead.

`runtime` (does onnxruntime load and run the exported graph) and
`numerics` (do its outputs match PyTorch) are reported as separate columns
on purpose. A graph that exports but diverges numerically is usually a
property of the kernel that ran, not a missing operator, so folding either
into the support glyph would blame the exporter for a runtime issue.

`blocked` and no-spec rows exist for the same reason: both are cases
where we have no evidence about ONNX, and reporting "no evidence" as ❌
would overstate what was measured.

## Measuring what we cannot translate

The catalog exists to guard our own exporter, so for a long time a spec
only existed where t2n succeeds. That gave the ONNX column a selection
bias with a precise shape: **the measured population was a subset of our
own supported set**. An operator ONNX handles and we do not could never
be graded ✅, only inherited as a retired ONNX listing claim, or
left blank. The comparison was structurally blind exactly where we are
weakest, and the `Gap vs ONNX` filter could only ever surface operators
that the torch 2.8 listing happened to name.

Specs for operators with **no translation at all** close that, purely so
the sweep can measure them. They live in the same themed module as
everything else in their family (`linalg.py`, `shape.py`, `reductions.py`
and so on) and are marked with `OpSpec.nnef_gap`:

```python
# in tests/proptest/op_specs/shape.py, beside the shape ops we do ship
gap_spec(
    "masked_select",
    mask_st(torch.masked_select, "masked_select"),
    REASON_DATA_DEPENDENT,
)
```

The marker is **asserted, not trusted**. A spec that merely skipped the
tract driver would keep reporting its operator as unsupported long after
someone implemented it, and the page with it. So:

- the tract driver (`proptest` env) attempts the real export and checks
  it fails at the declared `stage`. Both "it works now" and "it fails
  somewhere else" are test failures, with the fix spelled out in the
  message. This is cheap: a missing emitter raises during translation,
  before tract is invoked.
- `tests/test_proptest_nnef_gap.py` runs in the **default** suite and
  cross-checks every `no-emitter` gap against the live emitter registry,
  so registering the operator fails a test in the fast job too.
- the ONNX sweep ignores the marker entirely and measures the spec like
  any other. That is the whole point of it existing.

`stage` records where the failure lands, and it is the first thing
someone planning to close the gap needs to know:

| Stage | Meaning |
| --- | --- |
| `no-emitter` | nothing is registered for the operator at all |
| `export-error` | an emitter refuses, or the pipeline raises earlier (the RNG factories are constant-folded before the lookup) |
| `tract-error` | NNEF is written and tract then declines to load or run it. Needs an emitter to exist, so the gap must also set `emitter_registered=True`, which inverts the registry check from "must be absent" to "must be present" |
| `raw-error` | the export crashed with something that is not a `T2NError`, so the user gets a bare `TypeError` instead of a message naming the operator. Always a bug on our side, whether or not we ever translate the operator |

Specs whose op draws from an RNG also set `nondeterministic=True`. Export
and runtime stay measured; only the numerics axis is skipped, since
comparing two independent draws would report the definition of the
operator rather than anything about the exporter.

## Adding coverage

An operator shows `missing spec` in the `spec coverage` column until some
spec claims it. To fix that, add or extend a spec in
`tests/proptest/op_specs/` and set its `aten_ops` to the operator name(s)
**as this page lists them**: the page drops `_`-prefixed identifiers and
merges in-place variants, so the trace name is not always the row name
(`conv2d` traces `aten::_convolution`).

If t2n cannot translate the operator, the spec still belongs in the
catalog: put it in the themed module it would live in once supported and
give it an `nnef_gap`, instead of leaving the row unmeasured. Placing it
by family rather than by support status is deliberate, so that
implementing the operator later means deleting one field rather than
moving the spec between files. Rows that get no spec at all are rare:
they are recorded in `op_specs/untranslated.py` with a reason and are
listed in the support page appendix, outside both comparison
denominators.

`tests/test_proptest_aten_attribution.py` traces every spec and checks its
declaration, so a spec that drifts onto a different operator fails there
rather than silently mis-attributing its grade.
