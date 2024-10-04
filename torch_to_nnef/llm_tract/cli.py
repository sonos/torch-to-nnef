""" Export any huggingface transformers LLM to tract NNEF

With options to compress it to Q4_0 and use float16

"""
import argparse
import typing as T

from torch_to_nnef.exceptions import TorchToNNEFInvalidArgument
from torch_to_nnef.llm_tract.compress import dynamic_load_registry
from torch_to_nnef.llm_tract.config import LlamaSLugs, OpenELMSlugs, PHISlugs
from torch_to_nnef.llm_tract.exporter import dump_llm
from torch_to_nnef.log import log
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme

LOGGER = log.getLogger(__name__)


def parser_cli(
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
                help="export dir path to dump tokenizer infos, model config.json, model.nnef.tgz",
            )

        parser.add_argument(
            "-s",
            "--model-slug",
            help=f"huggingface slug (web-page 'endpoint') to export by example ({slug_examples})",
        )
        parser.add_argument(
            "-f16",
            "--as-float16",
            action="store_true",
            help="float in 16 bits",
        )
        parser.add_argument(
            "--compression-registry",
            default="torch_to_nnef.llm_tract.compress.DEFAULT_COMPRESSION",
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
