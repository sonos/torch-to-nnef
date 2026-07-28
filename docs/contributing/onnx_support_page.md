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
modules covering 300+ `aten::` operators, so the support level can simply
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

A full sweep is ~370 specs x 25 examples, each one an export plus an
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

The numerator is read the same generous way, and this is deliberate: it
counts what we measured as `full` **plus** what the retired listing
claimed and no spec of ours has checked (`✅*`). Those operators are
unverified, but the reason they are unverified is a gap in *our* test
coverage, and scoring that against ONNX would understate a competing
exporter for our own shortfall. Rows with neither a measurement nor a
claim (`-`) stay out: crediting a claim is not the same as crediting
silence, and a measured `partial` or `none` overrides any claim.

The caption under the bars splits the two populations apart, so the
strict "of what we actually tested, how much passed" ratio is still one
line away.

## Reading a grade

The `export` column grades operator coverage only:

| Glyph | Meaning |
| --- | --- |
| ✅ `full` | every generated example exported |
| 🟡 `partial` | some examples exported, others raised. The artifact keeps the failing shapes/dtypes |
| ❌ `none` | no example exported, and the exporter refused at least one |
| ⚠️ `blocked` | `torch.export` could not capture the module, so the ONNX exporter never ran. **Not** an ONNX verdict |
| ✅\* claimed | no spec covers it, so we did **not** verify it, but the retired listing claimed it was supported |
| `-` `untested` | no spec covers it and nothing was ever claimed either way. **Not** unsupported |

Only ✅ / 🟡 / ❌ are measurements. **✅\* is an unverified historical
claim**: the [headline bars](#reading-the-headline-bars) count it, so an
operator we never wrote a spec for is not scored against ONNX, while the
measured breakdown in the caption excludes it. The state exists so the two
kinds of "we don't know" stay distinguishable, since an operator the old
listing called supported is a better bet (and a better candidate for a new
spec) than one nobody ever said anything about. Both are filterable
separately.

`runtime` (does onnxruntime load and run the exported graph) and
`numerics` (do its outputs match PyTorch) are reported as separate columns
on purpose. A graph that exports but diverges numerically is usually a
property of the kernel that ran, not a missing operator, so folding either
into the support glyph would blame the exporter for a runtime issue.

`blocked` and `untested` exist for the same reason: both are cases where
we have no evidence about ONNX, and reporting "no evidence" as ❌ would
overstate what was measured.

## Adding coverage

An operator shows `-` until some spec claims it. To fix that, add or
extend a spec in `tests/proptest/op_specs/` and set its `aten_ops` to the
operator name(s) **as this page lists them**: the page drops
`_`-prefixed identifiers and merges in-place variants, so the trace name
is not always the row name (`conv2d` traces `aten::_convolution`).

`tests/test_proptest_aten_attribution.py` traces every spec and checks its
declaration, so a spec that drifts onto a different operator fails there
rather than silently mis-attributing its grade.
