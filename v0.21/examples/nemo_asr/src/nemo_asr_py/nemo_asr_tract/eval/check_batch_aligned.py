"""Check batch inference of a NeMo ASR model return identical results.

Only work on Open ASR Leaderboard datasets for now.
"""

import inspect
import logging
import os
import typing as T
from pathlib import Path

import evaluate
import numpy as np
import torch
from nemo.collections.asr.models import ASRModel
from pydantic import BaseModel as PydanticModel
from rich.console import Console as RichConsole

from nemo_asr_tract import NemoAsrModel
from nemo_asr_tract.dataset import (
    AUDIO_FILEPATHS_KEY,
    REFERENCES_KEY,
    DatasetConfig,
    collect_dataset,
    ensure_cache_dir,
    prepare_dataset,
    sort_by_duration,
)
from nemo_asr_tract.eval.compare_manifest import compute_wer, render_alignment
from nemo_asr_tract.eval.open_asr_leaderboard import (
    ESB_DATASETS,
    HF_ESB_SLUG,
    init_log,
)
from nemo_asr_tract.nemo_asr import RuntimeConfig
from nemo_asr_tract.normalizer.normalizer import EnglishTextNormalizer

NORMALIZER = EnglishTextNormalizer()


class Console:
    def __init__(self, console: T.Optional[RichConsole] = None):
        if console is None:
            console = RichConsole()
        self.console = console

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        # Additionally log to file if needed (e.g., using logging module)
        log_line = " ".join(str(arg) for arg in args)
        log_line = (
            log_line.replace("[red]", "")
            .replace("[green]", "")
            .replace("[bold underline]", "")
            .replace("[bold]", "")
            .replace("[/]", "")
        ).strip()
        logging.info(log_line)


class ErrorCase(PydanticModel):
    dataset: str
    split: str
    sample_idx: int
    reference: str
    prediction_nemo: str
    prediction_exported_unbatched: str
    prediction_exported_batched: str

    def is_an_error(self) -> bool:
        return not (
            NORMALIZER(self.prediction_nemo)
            == NORMALIZER(self.prediction_exported_batched)
            == NORMALIZER(self.prediction_exported_unbatched)
        )

    def wer_different_batched(self) -> float:
        return compute_wer(
            self.prediction_nemo,
            self.prediction_exported_batched,
            evaluate.load("wer"),
        )

    def wer_different_unbatched(self) -> float:
        return compute_wer(
            self.prediction_nemo,
            self.prediction_exported_unbatched,
            evaluate.load("wer"),
        )

    def print(self, console):
        console.print("_" * 80)
        console.print(
            f"[bold]Dataset:[/] {self.dataset} ([bold]{self.split}[/]), "
            f"[bold]sample index:[/] {self.sample_idx}"
        )
        console.print()

        def str_cond(cond) -> str:
            return "[green]ALIGNED[/]" if cond else "[red]MISALIGNED[/]"

        batched_ok = str_cond(self.wer_different_batched() == 0)
        unbatched_ok = str_cond(self.wer_different_unbatched() == 0)
        console.print(f"Reference:\n{NORMALIZER(self.reference)}")
        console.print()
        console.print(f"NeMo prediction:\n{NORMALIZER(self.prediction_nemo)}")
        console.print()
        console.print(f"Exported unbatched prediction ({unbatched_ok}):")
        console.print(
            render_alignment(
                NORMALIZER(self.prediction_nemo),
                NORMALIZER(self.prediction_exported_unbatched),
            )
        )
        console.print()
        console.print(f"Exported batched prediction ({batched_ok}):")
        console.print(
            render_alignment(
                NORMALIZER(self.prediction_nemo),
                NORMALIZER(self.prediction_exported_batched),
            )
        )


TensorLike = T.Union[torch.Tensor]


def _to_numpy(x: T.Any):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return None


class ModuleNpyDumper:
    """Attach post forward to a nn.Module to dump io as np arrays."""

    def __init__(
        self,
        module: torch.nn.Module,
        name: str,
        out_dir: T.Union[str, Path],
        dump_inputs: bool = True,
        dump_outputs: bool = True,
        once: bool = False,
    ):
        self.module = module
        self.name = name
        self.out_dir = out_dir
        self.dump_inputs = dump_inputs
        self.dump_outputs = dump_outputs
        self.once = once

        self.step = 0
        self._enabled = True

        os.makedirs(out_dir, exist_ok=True)

        self._post_handle = module.register_forward_hook(
            self._post_hook, with_kwargs=True
        )

    def _post_hook(
        self,
        module: torch.nn.Module,
        args: T.Tuple[T.Any, ...],
        kwargs: T.Dict[str, T.Any],
        outputs: T.Tuple[T.Any, ...],
    ):
        if not self._enabled:
            return

        if self.dump_inputs:
            self._dump_tensors(args, f"{self.name}_input/tensor")
            sig = inspect.signature(module.forward).bind(*args, **kwargs)

            for k, v in kwargs.items():
                # get index of the argument in forward signature
                ix = list(sig.arguments.keys()).index(k)
                self._dump_tensors(
                    v, f"{self.name}_inputs/tensor_{ix}_{self.step}"
                )

        if self.dump_outputs:
            self._dump_tensors(outputs, f"{self.name}_outputs/tensor")

        self.step += 1

        if self.once:
            self.disable()

    def _dump_tensors(self, outputs: T.Any, prefix: str):
        if isinstance(outputs, torch.Tensor):
            path = os.path.join(self.out_dir, f"{prefix}.npy")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.save(
                path,
                outputs.detach().cpu().numpy(),
            )
        elif isinstance(outputs, (tuple, list)):
            for i, out in enumerate(outputs):
                arr = _to_numpy(out)
                if arr is not None:
                    path = os.path.join(
                        self.out_dir,
                        f"{prefix}_{i}_{self.step}.npy",
                    )
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(
                        path,
                        arr,
                    )
        elif isinstance(outputs, dict):
            for k, out in outputs.items():
                arr = _to_numpy(out)
                if arr is not None:
                    # TODO: align notation with dirs of other types
                    np.save(
                        os.path.join(
                            self.out_dir, f"{prefix}_{k}_{self.step}.npy"
                        ),
                        arr,
                    )

    def disable(self):
        self._enabled = False

    def remove(self):
        self._post_handle.remove()


def use_pytorch_sdpa(model: torch.nn.Module):
    """Modify the model to use PyTorch sdpa implementations where applicable.

    This leverage attention modules set in NeMo with
    specific use_pytorch_sdpa flag.
    """
    # pylint: disable=import-outside-toplevel
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        MultiHeadAttention,
    )

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.use_pytorch_sdpa = True


class BatchBugReproducer:
    def __init__(
        self,
        model_dir: T.Union[str, Path],
        dump_io_path: T.Optional[T.Union[str, Path]] = None,
        force_cpu: bool = False,
        force_sdpa_pytorch: bool = False,
    ):
        self.model_dir = Path(model_dir)
        config = RuntimeConfig(
            max_n_tokens_per_step=10,
            force_cpu=force_cpu,
            encoder_per_batch=False,
        )
        if dump_io_path is not None:
            config.dump_intermediate_io_path = Path(dump_io_path) / "unbatched"
        self.model_wo_batch = NemoAsrModel.from_dir_with_runtime_config(
            model_dir, config
        )
        config.encoder_per_batch = True
        if dump_io_path is not None:
            config.dump_intermediate_io_path = Path(dump_io_path) / "batched"
        self.model_w_batch = NemoAsrModel.from_dir_with_runtime_config(
            model_dir, config
        )

        original_model = ASRModel.from_pretrained(
            self.model_w_batch.config.pretrained_name, map_location="cpu"
        )
        if dump_io_path is not None:
            self._nemo_encoder_dumper = ModuleNpyDumper(
                original_model.encoder,
                "encoder",
                Path(dump_io_path) / "nemo",
            )
        if force_sdpa_pytorch:
            use_pytorch_sdpa(original_model)
        self.original_model = original_model

    def get_data(
        self, config: DatasetConfig
    ) -> T.Tuple[T.List[str], T.List[str]]:
        cache_dir = ensure_cache_dir(config.name, config.split)
        dataset = prepare_dataset(config, cache_dir)
        all_data = sort_by_duration(collect_dataset(dataset))
        return all_data[AUDIO_FILEPATHS_KEY], all_data[REFERENCES_KEY]

    def run_inferences(
        self,
        dataset_config: DatasetConfig,
        wav_ix: int,
        generate_big_batch: bool = False,
    ) -> T.List[ErrorCase]:
        all_wav_paths, refs = self.get_data(dataset_config)
        batch_size = dataset_config.batch_size
        error_cases: T.List[ErrorCase] = []
        start_ix = (wav_ix // batch_size) * batch_size
        end_ix = min(start_ix + batch_size, len(all_wav_paths))
        batch = all_wav_paths[start_ix:end_ix]
        print("Running inference on batch size:", len(batch))
        original_transcriptions = self.original_model.transcribe(
            batch,
            batch_size=batch_size,
            num_workers=0,
        )
        safe_transcriptions = self.model_wo_batch.infer_from_wav_paths(batch)
        fail_transcriptions = self.model_w_batch.infer_from_wav_paths(batch)
        if generate_big_batch:
            _ = self.model_w_batch.infer_from_wav_paths(
                batch + [batch[0]] * (batch_size // 2)
            )
            _ = self.model_w_batch.infer_from_wav_paths(
                batch + [batch[0]] * batch_size
            )
            _ = self.model_w_batch.infer_from_wav_paths(
                batch + [batch[0]] * batch_size * 2
            )

        for (
            ix,
            ref,
            nemo_transcript,
            tract_unbatched_transcript,
            tract_batch_transcript,
        ) in zip(
            range(start_ix, end_ix),
            refs[start_ix:end_ix],
            original_transcriptions[0],
            (_.text for _ in safe_transcriptions),
            (_.text for _ in fail_transcriptions),
        ):
            err_case = ErrorCase(
                dataset=dataset_config.name,
                split=dataset_config.split,
                sample_idx=ix,
                reference=ref,
                prediction_nemo=nemo_transcript,
                prediction_exported_unbatched=tract_unbatched_transcript,
                prediction_exported_batched=tract_batch_transcript,
            )
            if err_case.is_an_error():
                error_cases.append(err_case)
        return error_cases


def make_length_mask(
    a: np.ndarray,
    lengths: np.ndarray,
    *,
    len_dim: int = 1,
) -> np.ndarray:
    """Create a broadcast-safe mask for variable-length tensors.

    Args:
        a : np.ndarray
            Input tensor of shape (B, ...).
        lengths : np.ndarray
            Lengths per batch element, shape (B,).
        len_dim : int
            Dimension to apply the length mask on.

    Returns:
        mask : np.ndarray
            Boolean mask with same rank as `a`, broadcastable to `a`.
    """
    assert lengths.ndim == 1
    assert a.shape[0] == lengths.shape[0]
    assert 0 <= len_dim < a.ndim

    B = a.shape[0]
    L = a.shape[len_dim]

    # Build base arange mask on len_dim
    arange = np.arange(L)

    # Reshape to (1, ..., 1, L, 1, ..., 1)
    shape = [1] * a.ndim
    shape[len_dim] = L
    arange = arange.reshape(shape)

    # Broadcast lengths to full rank
    len_shape = [1] * a.ndim
    len_shape[0] = B
    lengths_b = lengths.reshape(len_shape)

    base_mask = arange < lengths_b  # (B, ..., L, ...)

    # Explicitly expand all other dimensions to a.shape
    mask = np.broadcast_to(base_mask, a.shape)

    return mask


def compare_tensors(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    lengths: T.Optional[np.ndarray] = None,
    len_dim: int = 2,
    per_sample: bool = False,
    console: T.Optional[Console] = None,
    color_tol: float = 1e-5,
):
    if console is None:
        console = Console()
    if a.shape != b.shape:
        console.print(
            f"[red]Shape mismatch[/] for {name}: {a.shape} vs {b.shape}"
        )
        return

    if lengths is not None and len(a.shape) > 1:
        assert len(lengths) == a.shape[0] == b.shape[0], (
            "Batch size mismatch with lengths"
        )
        masks = make_length_mask(a, lengths, len_dim=len_dim)
        a = np.where(masks, a, 0)
        b = np.where(masks, b, 0)
    diff = np.abs(a - b)

    def display_diff_stats(
        name, max_d, mean_d, sample_idx=None, color_tol=1e-5
    ):
        if sample_idx is not None:
            name = f"{name} (sample {sample_idx})"
        max_d_str = f"{max_d:.2e}"
        mean_d_str = f"{mean_d:.2e}"

        def color_str(value, threshold):
            if value > threshold:
                return f"[red]{value:.2e}[/]"
            else:
                return f"[green]{value:.2e}[/]"

        max_d_str = color_str(max_d, color_tol)
        mean_d_str = color_str(mean_d, color_tol)

        console.print(
            f"[bold]{name}[/]: max|Δ|={max_d_str}, mean|Δ|={mean_d_str}"
        )

    if per_sample:
        max_diff_per_sample = diff.reshape(diff.shape[0], -1).max(axis=1)
        mean_diff_per_sample = diff.reshape(diff.shape[0], -1).mean(axis=1)
        for i, (max_d, mean_d) in enumerate(
            zip(max_diff_per_sample, mean_diff_per_sample)
        ):
            if max_d > 0 or mean_d > 0:
                display_diff_stats(
                    name, max_d, mean_d, sample_idx=i, color_tol=color_tol
                )
    else:
        max_d = diff.max()
        mean_d = diff.mean()
        display_diff_stats(name, max_d, mean_d, color_tol=color_tol)


def analyze_npy_dumps(
    dump_dir: T.Union[str, Path],
    generate_big_batch: bool = False,
    assume_io_one_is_lengths: bool = True,
    batch_size: int = 32,
    console: T.Optional[Console] = None,
):
    """Analyze dumped npy files to find potential causes of the batch bug.

    check absolute difference max/mean between unbatched and
    batched intermediate tensors, especially for encoder outputs
    (still check inputs are aligned).

    if generate_big_batch is enabled,
    also check if the difference becomes larger when batch size goes beyond 32,
    by comparing results to the big batch with duplicated samples
    (by removing the duplicate).

    """
    if console is None:
        console = Console()
    dump_dir = Path(dump_dir)

    def load_npy(path: Path) -> np.ndarray:
        return np.load(path)

    console.print("[bold]Analyzing dumped intermediate tensors[/]\n")

    # 1. Compare encoder inputs
    console.print("[bold underline]Encoder inputs (batched vs unbatched)[/]")
    ub_in_dir = dump_dir / "unbatched" / "encoder_inputs"
    b_in_dir = dump_dir / "batched" / "encoder_inputs"
    nemo_in_dir = dump_dir / "nemo" / "encoder_inputs"

    for ub_file in sorted(ub_in_dir.glob("*.npy")):
        b_file = b_in_dir / ub_file.name
        if not b_file.exists():
            console.print(f"[yellow]Missing batched input:[/] {ub_file.name}")
            continue

        ub = load_npy(ub_file)
        b = load_npy(b_file)
        lens = None
        if assume_io_one_is_lengths:
            lens = load_npy(ub_file.parent / "tensor_1_0.npy")
        compare_tensors(
            f"encoder_inputs/{ub_file.name}",
            ub,
            b,
            lengths=lens,
            console=console,
        )

    # 2. Compare encoder outputs
    console.print("\n[bold underline]Encoder outputs (batched vs unbatched)[/]")
    ub_out_dir = dump_dir / "unbatched" / "encoder_outputs"
    b_out_dir = dump_dir / "batched" / "encoder_outputs"
    nemo_out_dir = dump_dir / "nemo" / "encoder_outputs"

    for ub_file in sorted(ub_out_dir.glob("*.npy")):
        b_file = b_out_dir / ub_file.name
        if not b_file.exists():
            console.print(f"[yellow]Missing batched output:[/] {ub_file.name}")
            continue

        ub = load_npy(ub_file)
        b = load_npy(b_file)
        lens = None
        if assume_io_one_is_lengths:
            lens = load_npy(ub_file.parent / "tensor_1_0.npy")
        compare_tensors(
            f"encoder_outputs/{ub_file.name}",
            ub,
            b,
            lengths=lens,
            console=console,
        )
        compare_tensors(
            " > ",
            ub,
            b,
            lengths=lens,
            console=console,
            per_sample=True,
        )

    # 3. Compare against original NeMo encoder (sanity check)
    nemo_dir = dump_dir / "nemo"
    if nemo_dir.exists():
        console.print(
            "\n[bold underline]NeMo encoder vs exported (unbatched)[/]"
        )
        for ub_file in sorted(ub_in_dir.glob("*.npy")):
            # convention: nemo_encoder_step_0_output_{i}.npy
            idx = ub_file.stem.split("_", maxsplit=1)[-1]
            nemo_file = nemo_in_dir / ub_file.name

            if not nemo_file.exists():
                continue

            nemo = load_npy(nemo_file)
            ub = load_npy(ub_file)
            lens = None
            if assume_io_one_is_lengths:
                lens = load_npy(ub_file.parent / "tensor_1_0.npy")
            compare_tensors(
                f"nemo_vs_unbatched/input_{idx}",
                nemo,
                ub,
                lengths=lens,
                console=console,
            )

        for ub_file in sorted(ub_out_dir.glob("*.npy")):
            # convention: nemo_encoder_step_0_output_{i}.npy
            idx = ub_file.stem.split("_", maxsplit=1)[-1]
            nemo_file = nemo_out_dir / ub_file.name

            if not nemo_file.exists():
                continue

            nemo = load_npy(nemo_file)
            ub = load_npy(ub_file)
            lens = None
            if assume_io_one_is_lengths:
                lens = load_npy(ub_file.parent / "tensor_1_0.npy")
            compare_tensors(
                f"nemo_vs_unbatched/output_{idx}",
                nemo,
                ub,
                lengths=lens,
                console=console,
            )

    # 4. Optional: big batch analysis
    if generate_big_batch:
        console.print(
            f"\n[bold underline]Big batch analysis (sensitivity on the {batch_size} first samples of in batches)[/]"
        )

        for ref_ix in range(3):
            if ref_ix > 0:
                console.print("-" * 40)
            console.print(
                f"* Comparing reference batch (*_{ref_ix}.npy) vs other batch (*_n.npy) "
            )
            for base_file in sorted(b_out_dir.glob(f"*_{ref_ix}.npy")):
                other_ix = 0
                other_file = b_out_dir / base_file.name.replace(
                    f"_{ref_ix}.npy", f"_{other_ix}.npy"
                )
                while other_file.exists():
                    if ref_ix != other_ix:
                        base = load_npy(base_file)
                        other = load_npy(other_file)

                        if base.ndim == 0 or other.ndim == 0:
                            continue

                        # Slice other batch to match original batch size
                        base_batch_size = base.shape[0]
                        other_batch_size = other.shape[0]
                        base = base[:batch_size]
                        other_sliced = other[:batch_size]
                        lens = None
                        if assume_io_one_is_lengths:
                            lens = load_npy(base_file.parent / "tensor_1_0.npy")

                        compare_tensors(
                            f"base_{base_batch_size}_vs_other_{other_batch_size}/{base_file.name}",
                            base,
                            other_sliced,
                            lengths=lens,
                            console=console,
                        )
                        if (
                            ref_ix == 0
                        ):  # only compare with big batch when reference is the original batch
                            compare_tensors(
                                " > ",
                                base,
                                other_sliced,
                                lengths=lens,
                                console=console,
                                per_sample=True,
                            )
                    other_ix += 1
                    other_file = b_out_dir / base_file.name.replace(
                        f"_{ref_ix}.npy", f"_{other_ix}.npy"
                    )

    console.print("\n[green]Analysis complete.[/]")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the directory with the exported NeMo ASR model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=set([_[0] for _ in ESB_DATASETS]),
        help="Dataset to use for reproduction.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=set([_[1] for _ in ESB_DATASETS]),
        default="test",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        required=True,
        help="Index of the sample to reproduce the bug on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for reproduction.",
    )
    parser.add_argument(
        "--generate-big-batch",
        action="store_true",
        help="Whether to generate a big batch with duplicated samples to "
        "observe given dumps if behavior is related to batch size (beyond 32). "
        "by comparing results to batched model",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Whether to force CPU inference for reproduction. "
        "Set this if you want to check if the batch bug is related "
        "to GPU kernels.",
    )
    parser.add_argument(
        "--force-sdpa-pytorch",
        action="store_true",
        help="Whether to force using PyTorch sdpa implementation "
        "for attention. Set this if you want to compare with "
        "another nemo/Python implementation of attention .",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="If specified, dumps intermediate I/O tensors to the given path.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Logging verbosity level: -1 (ERROR), 0 (INFO), 1 (DEBUG).",
    )
    return parser.parse_args()


def main():
    """Idxes are collected from.

    nemo_tract_eval_compare_manifest \
        --results-dir $HOME/SONOS/data/dump_parakeet_test_libri_batched_new_model/librispeech/test.clean \
        --max-items 3

    """
    args = parse_args()
    init_log(
        verbosity=args.verbosity,
        log_file=(Path(args.output_dir) / "reproduction.log")
        if args.output_dir
        else None,
        disable_stdout=True,
    )
    # ix: 2489, 1784 in librispeech test.clean
    wav_ix = args.sample_idx
    batch_size = args.batch_size  # Batch size to reproduce the bug
    dataset_name = args.dataset
    dataset_split = args.split
    console = Console()
    console.print(f"Preparing dataset {dataset_name} split {dataset_split}...")
    dataset_config = DatasetConfig(
        hg_path=HF_ESB_SLUG,
        name=dataset_name,
        split=dataset_split,
        batch_size=batch_size,
        max_eval_samples=None,
        streaming=False,
    )

    reproducer = BatchBugReproducer(
        args.model_dir, args.output_dir, args.force_cpu, args.force_sdpa_pytorch
    )
    console.print(f"start evaluating...: {wav_ix}th wav file in the dataset")
    error_cases = reproducer.run_inferences(
        dataset_config,
        wav_ix=wav_ix,
        generate_big_batch=args.generate_big_batch,
    )
    if error_cases:
        for err in error_cases:
            err.print(console)
        console.print("_" * 80)
        console.print(f"\nTotal error cases: {len(error_cases)}")
        if args.output_dir is not None:
            console.print(
                f"Intermediate I/O tensors are dumped to: {args.output_dir}"
            )
            with open(args.output_dir / "error_cases.jsonl", "w") as f:
                for err in error_cases:
                    f.write(err.model_dump_json() + "\n")
            analyze_npy_dumps(
                Path(args.output_dir),
                generate_big_batch=bool(args.generate_big_batch),
                assume_io_one_is_lengths=True,
                batch_size=batch_size,
                console=console,
            )
    else:
        console.print("No error cases found 😊.")


if __name__ == "__main__":
    main()
