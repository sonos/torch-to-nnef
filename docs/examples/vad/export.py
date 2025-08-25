"""Simple export script of MarbleNet VAD."""

import argparse
import copy
import logging
import subprocess
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from omegaconf import OmegaConf

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.log import init_log


def parser_cli(
    description=__doc__,
):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Output directory for nnef assets",
    )

    parser.add_argument(
        "-t",
        "--tract-path",
        required=True,
        help="Tract cli binary path to allow pulsification",
    )

    parser.add_argument(
        "-s",
        "--vad-slug",
        default="vad_multilingual_marblenet",
        help="VAD slug to export from NeMo",
    )

    parser.add_argument(
        "-p",
        "--pulse",
        default=1600,
        help="Pulse in number of frames "
        "(aka frequency at which encoder will run tract inference),"
        " 1600 frames -> 10ms",
    )
    return parser.parse_args()


# sample rate, Hz
SAMPLE_RATE = 16000


class DummyDecoder(torch.nn.Module):
    def forward(self, encoder_output, **kwargs):
        return encoder_output


class EncoderWrapper(torch.nn.Module):
    """Avoid to expose input_len (that doesn't make sense in streaming)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        dim1 = torch.tensor(x.size(1)).repeat(x.size(0))
        return self.model(x, dim1)


def export(
    tract_path: Path,
    dump_path: Path,
    vad_slug: str = "vad_multilingual_marblenet",
    return_softmax: bool = True,
    pulse_value=1600,  # 10ms
):
    vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
        vad_slug
    )

    cfg = copy.deepcopy(vad_model._cfg)
    print(OmegaConf.to_yaml(cfg))

    vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)

    # Set model to inference mode
    vad_model.eval()
    vad_model = vad_model.to(vad_model.device)

    vad_model.preprocessor.featurizer.pad_to = (
        0  # in streaming this is important
    )

    decoder = vad_model.decoder
    decoder._return_logits = (
        not return_softmax  # return softmaxed output (easier to threshold from)
    )
    # split featurizer+encoder and decoder to benefit from tract streaming cache capacity on encoder
    vad_model.decoder = DummyDecoder()

    enc_path_export = dump_path / "vad_marblenet.encoder.nnef.tgz"
    export_model_to_nnef(
        model=EncoderWrapper(vad_model),  # any nn.Module
        args=(torch.rand(1, 512)),
        file_path_export=enc_path_export,
        inference_target=TractNNEF(
            version=TractNNEF.latest_version(),
            check_io=True,
            dynamic_axes={"input_signal": {0: "B", 1: "S"}},
            check_io_tolerance=TractCheckTolerance.SUPER,
        ),
        input_names=["input_signal", "input_len"],
        output_names=["output"],
        debug_bundle_path=dump_path / "debug_encoder.tgz",
        custom_extensions=[
            "tract_assert S > 1",
            "tract_assert B > 0",
        ],
    )

    dec_path_export = dump_path / "vad_marblenet.decoder.nnef.tgz"
    export_model_to_nnef(
        model=decoder,
        args=(torch.rand(2, 128, 4),),
        file_path_export=dec_path_export,  # filepath to dump NNEF archive
        inference_target=TractNNEF(
            version=TractNNEF.latest_version(),
            check_io=True,
            dynamic_axes={"encoder_output": {0: "B", 2: "S"}},
        ),
        input_names=["encoder_output"],
        output_names=["output"],
        debug_bundle_path=dump_path / "debug_decoder.tgz",
        custom_extensions=[
            "tract_assert S > 1",
            "tract_assert B > 0",
        ],
    )

    cmd = [
        str(tract_path.absolute()),
        str(enc_path_export.absolute()),
        "--nnef-tract-core",
        "--nnef-tract-pulse",
        f"--pulse S={pulse_value}",
        "dump",
        "--nnef",
        str(
            (
                dump_path
                / f"vad_marblenet.encoder.pulse_{pulse_value}.nnef.tgz"
            ).absolute()
        ),
    ]
    logging.info(" ".join(cmd))
    subprocess.check_output(cmd)
    logging.info("sucessful exports")


def main():
    init_log()
    args = parser_cli()
    export(
        dump_path=Path(args.output_path),
        tract_path=Path(args.tract_path),
        vad_slug=args.vad_slug,
        pulse_value=args.pulse,
    )


if __name__ == "__main__":
    main()
