"""Export any huggingface transformers LLM to tract NNEF

With options to compress it to Q4_0 and use float16

"""

import argparse
import typing as T
import logging

from torch_to_nnef.exceptions import TorchToNNEFInvalidArgument
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.compress import dynamic_load_registry
from torch_to_nnef.llm_tract.config import (
    DtypeStr,
    LlamaSLugs,
    OpenELMSlugs,
    PHISlugs,
)
from torch_to_nnef.llm_tract.exporter import dump_llm
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.log import init_log

LOGGER = logging.getLogger(__name__)


def parser_cli(  # pylint: disable=too-many-positional-arguments
    fn_parser_adder: T.Optional[
        T.Callable[[argparse.ArgumentParser], None]
    ] = None,
    description=__doc__,
    with_dump_with_tokenizer_and_conf: bool = True,
    with_test_display_token_gens: bool = True,
    # usefull to set False if just use `prep_exporter`
    with_export_args: bool = True,
    other_model_id_args: T.Optional[T.List[T.Tuple[str, str]]] = None,
):
    loader_parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    full_parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    slug_examples = ", ".join(
        [
            f"'{_.value}'"
            for slugsEnums in [LlamaSLugs, PHISlugs, OpenELMSlugs]
            for _ in slugsEnums
        ]
    )
    for parser in [loader_parser, full_parser]:
        if with_export_args:
            parser.add_argument(
                "-e",
                "--export-dirpath",
                required=True,
                help="export dir path to dump tokenizer infos, model "
                "config.json, model.nnef.tgz",
            )

        parser.add_argument(
            "-s",
            "--model-slug",
            help="huggingface slug (web-page 'endpoint') to "
            f"export by example ({slug_examples})",
        )
        parser.add_argument(
            "-dt",
            "--force-module-dtype",
            choices=[ds.value for ds in DtypeStr],
            help="apply model = model.to(force_dtype) ."
            " If force-module-dtype is float16 or bfloat16 "
            " flag `input-as-float16` is automatically applied."
            " If force-module-dtype is unset no `.to` will be applied "
            "(which may be wished for mixed-precision model)",
        )
        parser.add_argument(
            "-idt",
            "--force-inputs-dtype",
            choices=[ds.value for ds in DtypeStr],
            help="Force inputs float dtype (and ONLY the input)"
            "in case your model is custom mixed precision "
            "else we encourage you to use directly `--force-module-dtype`",
        )
        parser.add_argument(
            "--compression-registry",
            default="torch_to_nnef.compress.DEFAULT_COMPRESSION",
            help="Compression registry to load "
            "(should be a Dict[str, Callable(model, args)]), "
            "can be specified to load arbitrary compression library.",
        )
        parser.add_argument(
            "-d",
            "--local-dir",
            help="local dir containing .safetensors compatible with openELM"
            " model size specified in slug",
        )

        parser.add_argument(
            "-f32-attn",
            "--force-f32-attention",
            action="store_true",
            help="force f32 to happen in f32 even if f16 in rest of network",
        )

        parser.add_argument(
            "-f32-lin-acc",
            "--force-f32-linear-accumulator",
            action="store_true",
            help="force f32 to happen in linear(f16,f16)",
        )

        parser.add_argument(
            "-f32-norm",
            "--force-f32-normalization",
            action="store_true",
            help="force f32 to happen in all builtin torch normalization layers"
            "(batch_norm, norm, linalg_vector_norm, linalg_norm, layer_norm, "
            "group_norm, weight_norm)",
        )

        parser.add_argument(
            "-tt",
            "--tract-check-io-tolerance",
            default=TractCheckTolerance.APPROXIMATE.value,
            choices=[t.value for t in TractCheckTolerance],
            help="tract check io tolerance level",
        )

        if with_export_args:
            parser.add_argument(
                "-n",
                "--naming-scheme",
                default=VariableNamingScheme.NATURAL_VERBOSE_CAMEL.value,
                choices=[vns.value for vns in VariableNamingScheme],
                help="display debug information",
            )
            parser.add_argument(
                "--tract-specific-path",
                required=False,
                help="tract specific path (instead of latest version)",
            )
            parser.add_argument(
                "--tract-specific-version",
                required=False,
                help="tract specific version",
            )

        if with_test_display_token_gens:
            parser.add_argument(
                "-td",
                "--test-display-token-gens",
                action="store_true",
                help="Generate 50 tokens with model, "
                "and after f16/compression if activated "
                "this is meant as a way to detect spurious precision problems "
                "early",
            )

        if with_dump_with_tokenizer_and_conf and with_export_args:
            parser.add_argument(
                "-dwtac",
                "--dump-with-tokenizer-and-conf",
                action="store_true",
                help="dump tokenizer and conf at same dir as model",
            )
        parser.add_argument(
            "-sgts",
            "--sample-generation-total-size",
            type=int,
            default=6,
            help="Number of tokens to generate in total "
            "for reference 'modes' samples npz dumped ",
        )
        parser.add_argument(
            "-iaed",
            "--ignore-already-exist-dir",
            action="store_true",
            help="ignore already existing export dir",
        )
        parser.add_argument(
            "-nv",
            "--no-verify",
            action="store_true",
            help="skip all correctness checks of exported model "
            "also take opportunity to reduce RAM usage",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="display debug information",
        )
        if fn_parser_adder is not None:
            fn_parser_adder(parser)
    # == hack by using 1st parser without help to fill dynamically 2nd parser ==
    args, _ = loader_parser.parse_known_args()
    possible_compression_ids = list(
        dynamic_load_registry(args.compression_registry).keys()
    )
    parser = full_parser
    parser.add_argument(
        "-c",
        "--compression-method",
        choices=possible_compression_ids,
        help="possible compression method to apply on Model before export",
    )
    args = parser.parse_args()
    ref_model_id_args = [
        ("model_slug", "--model-slug"),
        ("local_dir", "--local-dir"),
    ] + (other_model_id_args or [])
    n_flags = sum(
        getattr(args, argname) is not None for argname, _ in ref_model_id_args
    )
    possible_model_id_args = ",".join(
        [f"'{cli_arg_name}'" for _, cli_arg_name in ref_model_id_args]
    )
    if n_flags == 0:
        raise TorchToNNEFInvalidArgument(
            f"You should provide one among: {possible_model_id_args}"
        )
    if n_flags > 1:
        raise TorchToNNEFInvalidArgument(
            f"You should only provide one of {possible_model_id_args}"
        )
    return args


def main():
    log = init_log()
    args = parser_cli()
    log_level = log.INFO
    if args.verbose:
        log_level = log.DEBUG
    kwargs = vars(args)
    del kwargs["verbose"]
    dump_llm(
        **kwargs,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
