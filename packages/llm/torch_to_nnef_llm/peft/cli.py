"""Export any PEFT weights (LORA, ...) from .pt, .bin, .safetensors to NNEF."""

import argparse
import json
import logging
import re
import subprocess
import tempfile
import typing as T
from collections import defaultdict
from pathlib import Path

import torch

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.export import (
    export_tensors_from_disk_to_nnef,
    iter_torch_tensors_from_disk,
)
from torch_to_nnef.log import init_log, set_lib_log_level
from torch_to_nnef.utils import cd

LOGGER = logging.getLogger(__name__)

NAME_PLACEHOLDER = "(?P<name>.*)"
DEFAULT_METHOD_TYPE = "LoRA"
PATTERN_LORA = [
    f"{NAME_PLACEHOLDER}.lora_A.default.weight$",
    f"{NAME_PLACEHOLDER}.lora_B.default.weight$",
]
MAP_TENSOR_NAME = "{name}.weight"
CHECK_MAP_TENSOR_NAME = "{name}.base_layer.weight"
PEFT_MAPPING_FILENAME = "peft_mapping.json"


def parser_cli(
    description=__doc__,
):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--read-filepath",
        required=True,
        help="file to read containing the tensors to extract "
        "(.pt, .pth, .bin, .safetensors)",
    )

    parser.add_argument(
        "-p",
        "--patterns",
        nargs="+",
        default=PATTERN_LORA,
        help="regex patterns capturing parameters name in nn.Module to extract "
        "PEFT matrices by torch naming, should contains: "
        f"{NAME_PLACEHOLDER} to map original matrix",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD_TYPE,
        help="PEFT method name",
    )
    parser.add_argument(
        "-c",
        "--check-map-name",
        default=CHECK_MAP_TENSOR_NAME,
        help="PEFT targeted base matrix name in PEFT nn.Module",
    )
    parser.add_argument(
        "-m",
        "--map-name",
        default=MAP_TENSOR_NAME,
        help="PEFT targeted base matrix name in original nn.Module",
    )
    parser.add_argument(
        "-o",
        "--output-archive",
        required=True,
        help="should finish by .peft.nnef.tgz",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display debug information",
    )
    return parser.parse_args()


# pylint: disable-next=too-many-positional-arguments
def export_peft(
    read_filepath: Path,
    output_archive: Path,
    patterns: T.Optional[T.List[str]] = None,
    map_name: str = MAP_TENSOR_NAME,
    check_map_name: str = CHECK_MAP_TENSOR_NAME,
    method_name: str = DEFAULT_METHOD_TYPE,
):
    if patterns is None:
        patterns = PATTERN_LORA
    LOGGER.info("start to export of PEFT tensors to NNEF")
    assert output_archive.name.endswith(".peft.nnef.tgz"), output_archive
    assert not output_archive.exists(), output_archive
    for pat in patterns:
        assert NAME_PLACEHOLDER in pat, pat

    jpattern = "|".join([p.replace(NAME_PLACEHOLDER, ".*") for p in patterns])

    def filter_key(key):
        matches = re.match(jpattern, key)
        LOGGER.debug("found '%s' match '%s' %s", key, jpattern, matches)
        return bool(matches)

    def fn_check_found_tensors(to_export):
        qte = len(to_export)
        if qte == 0:
            raise T2NErrorMisuse(
                f"no tensors found in provided file with pattern: {patterns}"
            )
        if qte % len(patterns) != 0:
            raise T2NErrorMisuse(
                f"Number of PEFT matrices found is {qte} "
                f"should be multiple of {len(patterns)}"
            )
        return True

    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        name_to_tensors = export_tensors_from_disk_to_nnef(
            store_filepath=read_filepath,
            output_dir=td,
            filter_key=filter_key,
            fn_check_found_tensors=fn_check_found_tensors,
        )
        mapping_table = defaultdict(list)
        to_check_tensors = set()
        for k, _ in name_to_tensors.items():
            prefix_tensor_name = None
            for pat in patterns:
                match = re.search(pat, k)
                if match:
                    prefix_tensor_name = match.group("name")
                    break
            if prefix_tensor_name is None:
                raise T2NErrorMisuse(k, patterns)
            ref_matrix_name = map_name.format(name=prefix_tensor_name)
            mapping_table[ref_matrix_name].append(k)
            to_check_tensors.add(check_map_name.format(name=prefix_tensor_name))
        expanded_ref_names = set(mapping_table.keys())
        found_ref_tensors: T.Dict[str, torch.Tensor] = dict(
            iter_torch_tensors_from_disk(
                read_filepath, lambda x: x in to_check_tensors
            )
        )
        LOGGER.info(
            "found PEFT applied for %d base tensors", len(found_ref_tensors)
        )
        if len(found_ref_tensors) != len(mapping_table):
            missing_ref_tensors = set(expanded_ref_names).difference(
                found_ref_tensors
            )
            raise T2NErrorMisuse(
                "missing following ref tensors in provided "
                f"read file: {missing_ref_tensors}"
            )

        with (td / PEFT_MAPPING_FILENAME).open("w", encoding="utf8") as fh:
            json.dump(
                # mapping table is for original model naming
                {"method": method_name, "mapping_table": mapping_table},
                fh,
                indent=2,
            )

        with cd(td):
            subprocess.check_call(["tar", "-czf", output_archive, "."])

        LOGGER.info(
            "successful export of PEFT tensors to NNEF: %s", output_archive
        )


def main():
    log = init_log()
    args = parser_cli()
    log_level = log.INFO
    if args.verbose:
        log_level = log.DEBUG
    set_lib_log_level(log_level)
    export_peft(
        read_filepath=Path(args.read_filepath),
        output_archive=Path(args.output_archive),
        patterns=args.patterns,
        map_name=args.map_name,
        check_map_name=args.check_map_name,
        method_name=args.method,
    )


if __name__ == "__main__":
    main()
