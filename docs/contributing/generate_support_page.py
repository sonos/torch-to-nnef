import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import bs4
import requests as rq

from torch_to_nnef.op.aten import aten_ops_registry

TORCH_VERSION = "v2.7.1"
URL_IR = "https://docs.pytorch.org/docs/main/torch.compiler_ir.html"
ONNX_SUPPORT_URL = "https://docs.pytorch.org/docs/stable/onnx_torchscript_supported_aten_ops.html"


def get_core_ir():
    resp = rq.get(URL_IR, timeout=20)
    assert resp.status_code == 200
    soup = bs4.BeautifulSoup(resp.content, "html.parser")
    res = soup.find_all("span", {"class": "pre"})
    official_aten_names = {
        r.text.split(".")[1]
        for r in res
        if r.text.startswith("aten")
        if "backward" not in r.text
    }
    official_prim_names = sorted(
        [r.text.split(".")[1] for r in res if r.text.startswith("prim")]
    )
    return (official_aten_names, official_prim_names)


def get_onnx_support():
    resp = rq.get(ONNX_SUPPORT_URL, timeout=20)
    assert resp.status_code == 200
    soup = bs4.BeautifulSoup(resp.content, "html.parser")
    supported_ops = {
        _.text.replace("aten::", "")
        for _ in soup.find(id="id1").find_all("span", {"class": "pre"})
        if "aten::" in _.text
    }
    unsupported_ops = {
        _.text.replace("aten::", "")
        for _ in soup.find(id="id2").find_all("span", {"class": "pre"})
        if "aten::" in _.text
    }
    return supported_ops, unsupported_ops


class LinkToTorchDocCache:
    UNK = "unk"

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_dic = self.load()

    def load(self):
        base = defaultdict(set)
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf8") as fh:
                for pat, elms in json.load(fh).items():
                    for elm in elms:
                        base[pat].add(elm)
        return base

    def save(self):
        with self.cache_path.open("w", encoding="utf8") as fh:
            json.dump(
                {k: sorted(list(v)) for k, v in self.cache_dic.items()},
                fh,
                indent=4,
            )

    def add(self, pattern: str, op_name: str, exclusive_pattern: bool = True):
        for k, v in self.cache_dic.items():
            if k is self.UNK:
                continue
            if op_name in v and exclusive_pattern:
                return
        if rq.get(pattern.format(op_name), timeout=20).status_code == 200:
            self.cache_dic[pattern].add(op_name)
            if op_name in self.cache_dic[self.UNK]:
                self.cache_dic[self.UNK].remove(op_name)
        else:
            self.cache_dic[self.UNK].add(op_name)

    def get_url(self, op_name) -> Optional[str]:
        for k, v in self.cache_dic.items():
            if op_name in v and k != self.UNK:
                return k.format(op_name)
        return None


official_aten_names, official_prim_names = get_core_ir()
t2n_aten = set(list(aten_ops_registry._registry.keys()))
onnx_supported, onnx_unsupported = get_onnx_support()

aten_torch_from_code = sorted(
    subprocess.check_output(
        "cd /tmp ; "
        "git clone -q git@github.com:pytorch/pytorch.git || git -C 'pytorch' pull; "
        "cd /tmp/pytorch ;"
        f"git checkout {TORCH_VERSION}; "
        'rg "aten::" | sed "s|.*aten::\\([a-zA-Z0-9_]*\\).*|\\1|g"|sort|uniq',
        shell=True,
    )
    .decode("utf8")
    .split("\n")
)
aten_torch_from_code = [
    _ for _ in aten_torch_from_code if not _.startswith("_")
]
aliases = sorted(
    subprocess.check_output(
        "cd /tmp ; "
        "git -C 'pytorch' pull || git clone -q git@github.com:pytorch/pytorch.git; "
        "cd /tmp/pytorch ;"
        f"git checkout {TORCH_VERSION}; "
        "cat ./torch/csrc/jit/passes/normalize_ops.cpp",
        shell=True,
    )
    .decode("utf8")
    .split("\n")
)
naliases = {
    tuple(x.replace("aten::", "") for x in a.strip()[1:-2].split(", "))
    for a in aliases
    if "{" in a and "}" in a and "aten::" in a
}
alias_map = {k: v for (k, v) in naliases}
ref_alias = defaultdict(list)
for k, v in alias_map.items():
    ref_alias[v].append(k)

support_inplace = set()
offset = 0
for ix, a in enumerate(aten_torch_from_code[:]):
    if (
        a.endswith("_")
        and a[:-1] in aten_torch_from_code
        or a in alias_map
        or a.strip() == ""
        or (len(a) and a[0].isupper())
        or "backward" in a
        or a.startswith("sym_")
    ):
        del aten_torch_from_code[ix - offset]
        offset += 1
        support_inplace.add(a[:-1])

cache_url = LinkToTorchDocCache(Path(__file__).parent / "torch_doc_urls.json")
for a_from_code in aten_torch_from_code:
    cache_url.add(
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.{}.html",
        a_from_code,
    )
    cache_url.add(
        "https://docs.pytorch.org/docs/stable/generated/torch.{}.html",
        a_from_code,
    )


def print_t(text, file):
    """Print tabbed"""
    if text:
        if "\n" in text:
            lines = text.split("\n")
            new_lines = []
            for line in lines:
                new_line = f"    {line}" if line.strip() else line
                new_lines.append(new_line)
            text = "\n".join(new_lines)
        else:
            text = f"    {text}"
        print(text, file=file)
    else:
        print("", file=file)


def write_operator_support(
    support_target_name: str, support_target_msg: str, supported_opset: Set[str]
):
    rows = []
    qte_core = 0
    qte_supported_core = 0
    matched_qte = 0

    print(f'=== "{support_target_name}"', file=fh)
    print("", file=fh)
    for a_from_code in aten_torch_from_code:
        if a_from_code in alias_map:
            continue
        is_core = a_from_code in official_aten_names
        is_core_official_str = "✅" if is_core else "-"

        exist_in_support = a_from_code in supported_opset

        if is_core:
            qte_core += 1
            if exist_in_support:
                qte_supported_core += 1

        mapped_in_support_str = "✅" if exist_in_support else "❌"
        if exist_in_support:
            matched_qte += 1

        inplace_str = "✅" if a_from_code in support_inplace else "❌"
        alias_str = ", ".join(sorted(ref_alias[a_from_code]))
        op_name = a_from_code
        torch_url_doc = cache_url.get_url(op_name)
        if torch_url_doc:
            op_name = f"[{op_name}]({torch_url_doc})"
        rows.append(
            (
                f"| {op_name} | {alias_str} | {inplace_str} | {is_core_official_str} | {mapped_in_support_str} |",
                is_core,
            )
        )
    rows = sorted(rows, key=lambda x: -int(x[1]))
    print_t("", file=fh)
    support_n_ops = len([_ for _ in supported_opset if not _.endswith("_")])
    ratio_total_str = f"{matched_qte}/{len(aten_torch_from_code)}"
    print_t(
        f"Total matched operators in {support_target_msg} compared to:\n\n"
        f"- core PyTorch opset:\n\n"
        f'[={qte_supported_core}/{qte_core} "{qte_supported_core}/{qte_core}"]\n\n'
        "-  and support from full `aten::`: \n\n"
        f'[={ratio_total_str} "{ratio_total_str}"]\n\n'
        " (total registered aten "
        f"operators in t2n being {support_n_ops})",
        file=fh,
    )
    print_t("", file=fh)
    print_t(
        "| aten name | aliases | can in-place | is core | translated |",
        file=fh,
    )
    print_t(
        "| -------- | ------- | ------- | --------- | ---------------- |",
        file=fh,
    )
    for r in rows:
        print_t(r[0], file=fh)

    print_t("", file=fh)


with (Path(__file__).parent / "./supported_operators.md").open(
    "w", encoding="utf8"
) as fh:
    date = datetime.now().strftime("%d %b %Y")
    print(
        "!!! note\n"
        f"    This table and page are auto generated from 'a script' that dig into PyTorch."
        f" Version targetted is:  **'{TORCH_VERSION}'**. file was generated the **{date}**.\n\n"
        "!!! warning\n"
        "     Take these informations with a grain of salt as this is referencing operators that may never appear"
        " in torch IR graph traced by `torch_to_nnef` (because remapped to others more generic). Also some "
        " uncommon operators are very rare in models, hence support may be lacking. "
        " **SONOS only maintains operators 'per need basis'**, but contributions are always wecome [see how](./add_new_aten_op.md)."
        "\n\n"
        f"\n 'is core' column refers to this [PyTorch IR documentation page]({URL_IR})\n\n"
        "We filter-out from from observed operators 'backward' and 'sym' one's which are unwanted in inference engine. Also in place operations are merged with memory allocated activations as this is inference implementation detail.",
        file=fh,
    )
    write_operator_support("TractNNEF", "`torch_to_nnef`", t2n_aten)
    write_operator_support(
        "ONNX",
        f"builtin PyTorch `ONNX` support based on [this page]({ONNX_SUPPORT_URL})",
        onnx_supported,
    )

cache_url.save()
