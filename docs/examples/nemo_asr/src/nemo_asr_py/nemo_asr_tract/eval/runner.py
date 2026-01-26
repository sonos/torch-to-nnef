"""Asr Runner Abstraction for evaluation system."""

import importlib
import logging
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List

from tqdm import tqdm

import torch
from nemo import __version__ as nemo_version
from nemo.collections.asr.models import ASRModel
from nemo_asr_tract import __version__ as nemo_asr_tract_version
from nemo_asr_tract.eval.conf import DecodingStragegy, EvalConfig
from nemo_asr_tract.nemo_asr import NemoAsrModel, load_config_from_dir

# =============================================================================
# Device / model
# =============================================================================


def setup_device(device_id: int):
    if device_id >= 0:
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            print(
                "CUDA is not available; using Apple Silicon GPU via MPS backend"
                "(if the runner support it, else fallback to CPU)."
            )
            return torch.device("mps"), torch.float32
        return torch.device(f"cuda:{device_id}"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def import_class_from_string(path: str):
    """Import a class from a fully-qualified string path.

    Example:
        cls = import_class_from_string(
            "nemo_asr_tract.open_asr_leaderboard.ExportedNemoRunner"
        )
    """
    module_path, class_name = path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class AsrRunner(ABC):
    """Abstract ASR runner interface.

    Used to encapsulate different ASR model running evaluation.
    (potentially from different inference backends)

    """

    def __init__(self, desc_suffix: str):
        self.desc_suffix = desc_suffix

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_eval_config(
        cls,
        *,
        cfg: EvalConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "AsrRunner":
        """Load the ASR runner from a model directory."""
        raise NotImplementedError

    @abstractmethod
    def transcribe_from_wav_paths(self, wav_paths: List[str]) -> List:
        raise NotImplementedError


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def clean_name(name: str) -> str:
    return (
        name.replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


class ExportedNemoRunner(AsrRunner):
    """Exported Nemo-Tract model ran with tract.

    Does NOT batch internally → chunking handled here.
    """

    def __init__(self, model: NemoAsrModel, batch_size: int):
        super().__init__(desc_suffix="with exported model")
        self.model = model
        self.batch_size = batch_size

    def name(self) -> str:
        return clean_name(
            f"tract_runner_v{nemo_asr_tract_version}_"
            f"{self.model.config.pretrained_name}"
        )

    @classmethod
    def load_from_eval_config(
        cls,
        *,
        cfg: EvalConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "AsrRunner":
        """Load the ASR runner from a model directory."""
        model = NemoAsrModel.from_dir(cfg.model_dir)
        if dtype != torch.float32:
            logging.warning(
                "Exported Nemo-Tract models provide not "
                "control for dtype; ignoring dtype=%s",
                dtype,
            )
        if cfg.decoding_stragegy != DecodingStragegy.GREEDY:
            raise NotImplementedError(
                "Nemo-Tract exported models currently only "
                "support greedy decoding."
            )
        return cls(model, batch_size=cfg.batch_size)

    def transcribe_from_wav_paths(self, wav_paths: List[str]):
        transcripts = []

        for batch in tqdm(
            chunks(wav_paths, self.batch_size),
            desc=self.desc_suffix,
            total=(len(wav_paths) + self.batch_size - 1) // self.batch_size,
        ):
            transcripts.extend(self.model.infer_from_wav_paths(batch))

        return transcripts


class NemoRunner(AsrRunner):
    """Standard Python NeMo runner.

    Handles batching internally via model.transcribe.
    Also supports Canary via pnc flag.
    """

    def __init__(
        self,
        model: ASRModel,
        infer_fn: Callable,
        pretrained_name: str,
        desc: str,
    ):
        super().__init__(desc_suffix=desc)
        self.pretrained_name = pretrained_name
        self._infer = infer_fn

    def name(self) -> str:
        return clean_name(f"nemo_v{nemo_version}_{self.pretrained_name}")

    @classmethod
    def load_from_eval_config(
        cls,
        *,
        cfg: EvalConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "NemoRunner":
        conf = load_config_from_dir(cfg.model_dir)
        model_id = conf.pretrained_name

        if model_id.endswith(".nemo"):
            model = ASRModel.restore_from(model_id, map_location=device)
        else:
            model = ASRModel.from_pretrained(model_id, map_location=device)

        model.to(dtype)
        model.eval()

        if cfg.decoding_stragegy == DecodingStragegy.GREEDY:
            model.cfg.decoding.beam.beam_size = 1
            # https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml#L158C16-L158C28
            # can be greedy, greedy_batch, beam, tsd, alsd.
            model.cfg.decoding.strategy = "greedy"
            model.cfg.decoding.greedy.max_symbols = 10
        else:
            raise NotImplementedError(
                f"Decoding strategy {cfg.decoding_stragegy} not supported yet."
            )
        model.change_decoding_strategy(model.cfg.decoding)

        kwargs = dict(
            batch_size=cfg.batch_size,
            verbose=False,
            num_workers=0,
        )

        desc = "with nemo"

        infer_fn = partial(model.transcribe, **kwargs)
        return cls(model, infer_fn, pretrained_name=model_id, desc=desc)

    def transcribe_from_wav_paths(self, wav_paths: List[str]):
        return self._infer(wav_paths)


# =============================================================================
# Transcription
# =============================================================================


def transcribe(
    runner: AsrRunner,
    audio_files: List[str],
):
    with torch.inference_mode():
        return runner.transcribe_from_wav_paths(audio_files)


def measure_transcription_time(
    runner: AsrRunner,
    audio_files: List[str],
    cfg: EvalConfig,
):
    transcriptions = []
    total_time = 0.0

    for i in range(cfg.warmup + 1):
        files = (
            audio_files[: cfg.batch_size * 4]
            if i == 0 and cfg.warmup > 0
            else audio_files
        )

        start = time.time()
        transcriptions = transcribe(runner, files)
        end = time.time()

        if i == cfg.warmup:
            total_time = end - start

    return total_time, transcriptions


def load_runner_from_config(cfg: EvalConfig) -> AsrRunner:
    RunnerCls = import_class_from_string(cfg.model_runner_class)
    assert issubclass(RunnerCls, AsrRunner), (
        f"Provided model runner class {RunnerCls} "
        "is not a subclass of AsrRunner"
    )
    device, dtype = setup_device(cfg.device_id)
    return RunnerCls.load_from_eval_config(
        cfg=cfg,
        device=device,
        dtype=dtype,
    )
