"""Check batch inference of a NeMo ASR model return identical results.

Only work on Open ASR Leaderboard datasets for now.
"""

from pathlib import Path
import typing as T
import os
import numpy as np

import torch
import evaluate
from nemo.collections.asr.models import ASRModel
from rich.console import Console
from pydantic import BaseModel as PydanticModel

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
from nemo_asr_tract.eval.open_asr_leaderboard import ESB_DATASETS, HF_ESB_SLUG
from nemo_asr_tract.nemo_asr import RuntimeConfig
from nemo_asr_tract.normalizer.normalizer import EnglishTextNormalizer

NORMALIZER = EnglishTextNormalizer()


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
            return "[green]OK[/]" if cond else "[red]ERROR[/]"

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
    """Attach pre/post forward to a nn.Module to dump io as np arrays."""

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

        self._post_handle = module.register_forward_hook(self._post_hook)

    def _post_hook(
        self,
        module: torch.nn.Module,
        inputs: T.Tuple[T.Any, ...],
        outputs: T.Tuple[T.Any, ...],
    ):
        if not self._enabled:
            return

        prefix = f"{self.name}_step_{self.step}"

        if self.dump_inputs:
            self._dump_tensors(inputs, f"{prefix}_input")

        if self.dump_outputs:
            self._dump_tensors(outputs, prefix + "_output")

        self.step += 1

        if self.once:
            self.disable()

    def _dump_tensors(self, outputs: T.Any, prefix: str):
        if isinstance(outputs, torch.Tensor):
            np.save(
                os.path.join(self.out_dir, f"{prefix}.npy"),
                outputs.detach().cpu().numpy(),
            )
        elif isinstance(outputs, (tuple, list)):
            for i, out in enumerate(outputs):
                arr = _to_numpy(out)
                if arr is not None:
                    np.save(
                        os.path.join(self.out_dir, f"{prefix}_{i}.npy"),
                        arr,
                    )
        elif isinstance(outputs, dict):
            for k, out in outputs.items():
                arr = _to_numpy(out)
                if arr is not None:
                    np.save(
                        os.path.join(self.out_dir, f"{prefix}_{k}.npy"),
                        arr,
                    )

    def disable(self):
        self._enabled = False

    def remove(self):
        self._pre_handle.remove()
        self._post_handle.remove()


class BatchBugReproducer:
    def __init__(
        self,
        model_dir: T.Union[str, Path],
        dump_io_path: T.Optional[T.Union[str, Path]] = None,
    ):
        self.model_dir = Path(model_dir)
        config = RuntimeConfig(
            max_n_tokens_per_step=10, force_cpu=False, encoder_per_batch=False
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
                "nemo_encoder",
                Path(dump_io_path) / "nemo",
            )
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
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="If specified, dumps intermediate I/O tensors to the given path.",
    )
    return parser.parse_args()


def main():
    """Idxes are collected from.

    nemo_tract_eval_compare_manifest \
        --results-dir $HOME/SONOS/data/dump_parakeet_test_libri_batched_new_model/librispeech/test.clean \
        --max-items 3

    """
    args = parse_args()
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

    reproducer = BatchBugReproducer(args.model_dir, args.output_dir)
    console.print(f"start evaluating...: {wav_ix}th wav file in the dataset")
    error_cases = reproducer.run_inferences(dataset_config, wav_ix=wav_ix)
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
    else:
        console.print("No error cases found 😊.")


if __name__ == "__main__":
    main()
