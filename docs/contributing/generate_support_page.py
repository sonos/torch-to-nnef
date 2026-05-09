"""Generate supported operators markdown page.

Allow to compare supported operators in `torch_to_nnef` and `ONNX` builtin
support against core PyTorch operators as per PyTorch IR documentation.

Disclaimer: this is a best effort script that may not reflect 100% reality
of operator support in all cases. It is meant to give a general idea
of the coverage level of `torch_to_nnef` regarding PyTorch operators.
"""

import argparse
import json
import re
import subprocess
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import bs4
import requests as rq
import rich.progress

from torch_to_nnef.op.aten import aten_ops_registry


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
            # `==` not `is`: the UNK key survives a JSON round-trip as
            # an equal-but-not-identical string, so `is self.UNK` lets
            # the UNK bucket short-circuit the early return on every
            # op (everything ends up in UNK on first probe), defeating
            # the purpose of additional URL-pattern fallbacks.
            if k == self.UNK:
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


class AliasManager:
    def __init__(self, alias_tups: Set[tuple[str, ...]]):
        self._alias_tups = alias_tups
        self._aliases = set([_[0] for _ in alias_tups])
        self.ref_alias = defaultdict(list)
        for k, v in self._alias_tups:
            self.ref_alias[v].append(k)

        # sorted aliases for consistent output
        for k, v in self.ref_alias.items():
            self.ref_alias[k] = sorted(v)

    def is_alias(self, op_name: str) -> bool:
        return op_name in self._aliases

    def get_aliases(self, op_name: str) -> List[str]:
        return self.ref_alias.get(op_name, [])


#: Last torch version whose ``torch.compiler_ir.html`` page enumerates the
#: core ATen IR ops in scrapeable form. Starting at 2.10 the page was
#: emptied (the published version is ~1 KB of boilerplate), so we fall
#: back to this one to keep the "is core" column populated.
LAST_TORCH_VERSION_WITH_IR_DOC = "2.9"


class FetchFromTorchVersion:
    def __init__(self, torch_version: str):
        self.torch_version = torch_version
        # Set by `get_core_ir`: the URL that actually yielded the core
        # IR list (fallback or not). Used by the markdown header so the
        # `is core` link points at the page that was scraped.
        self.resolved_ir_url: Optional[str] = None

    @property
    def url_ir(self) -> str:
        return f"https://docs.pytorch.org/docs/{self.torch_version}/torch.compiler_ir.html"

    @property
    def url_ir_fallback(self) -> str:
        return (
            "https://docs.pytorch.org/docs/"
            f"{LAST_TORCH_VERSION_WITH_IR_DOC}/torch.compiler_ir.html"
        )

    @property
    def onnx_support_url(self) -> str:
        return (
            f"https://docs.pytorch.org/docs/{self.torch_version}/"
            "onnx_torchscript_supported_aten_ops.html"
        )

    @staticmethod
    def _parse_ir_page(html: bytes) -> tuple[Set[str], List[str]]:
        soup = bs4.BeautifulSoup(html, "html.parser")
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
        return official_aten_names, official_prim_names

    def get_core_ir(self) -> tuple[Set[str], List[str]]:
        resp = rq.get(self.url_ir, timeout=20)
        assert resp.status_code == 200
        official_aten_names, official_prim_names = self._parse_ir_page(
            resp.content
        )
        self.resolved_ir_url = self.url_ir
        if not official_aten_names:
            warnings.warn(
                f"{self.url_ir} no longer enumerates the core ATen IR "
                "(emptied in torch 2.10+); falling back to "
                f"{self.url_ir_fallback} for the 'is core' column.",
                stacklevel=2,
            )
            fallback = rq.get(self.url_ir_fallback, timeout=20)
            assert fallback.status_code == 200
            official_aten_names, official_prim_names = self._parse_ir_page(
                fallback.content
            )
            self.resolved_ir_url = self.url_ir_fallback
        return official_aten_names, official_prim_names

    def get_onnx_support(self) -> tuple[Set[str], Set[str]]:
        """Fetch the TorchScript-ONNX per-op support page.

        PyTorch removed this page after torch 2.8 (TorchScript ONNX
        export was deprecated in favour of `torch.onnx.export(dynamo=
        True)`), and the new dynamo path doesn't ship a tabular
        per-op page. Return empty sets when the URL 404s so callers
        can drop the ONNX section gracefully.
        """
        resp = rq.get(self.onnx_support_url, timeout=20)
        if resp.status_code == 404:
            warnings.warn(
                f"ONNX support page not found at {self.onnx_support_url} "
                "(removed in torch 2.9+). Skipping ONNX comparison.",
                stacklevel=2,
            )
            return set(), set()
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

    def get_aten_torch_from_code(self) -> List[str]:
        aten_torch_from_code = sorted(
            subprocess.check_output(
                "cd /tmp ; "
                "git clone -q git@github.com:pytorch/pytorch.git || "
                "git -C 'pytorch' pull; "
                "cd /tmp/pytorch ;"
                f"git checkout v{self.torch_version}.0; "
                'rg "aten::" | '
                'sed "s|.*aten::\\([a-zA-Z0-9_]*\\).*|\\1|g"|sort|uniq',
                shell=True,
            )
            .decode("utf8")
            .split("\n")
        )
        return [_ for _ in aten_torch_from_code if not _.startswith("_")]

    def get_aliases_from_code(self) -> AliasManager:
        aliases = sorted(
            subprocess.check_output(
                "cd /tmp ; "
                "git -C 'pytorch' pull || "
                "git clone -q git@github.com:pytorch/pytorch.git; "
                "cd /tmp/pytorch ;"
                f"git checkout v{self.torch_version}.0; "
                "cat ./torch/csrc/jit/passes/normalize_ops.cpp",
                shell=True,
            )
            .decode("utf8")
            .split("\n")
        )
        return AliasManager(
            {
                tuple(
                    x.replace("aten::", "") for x in a.strip()[1:-2].split(", ")
                )
                for a in aliases
                if "{" in a and "}" in a and "aten::" in a
            }
        )

    def get_cache_url(
        self,
        aten_torch_from_code: List[str],
    ) -> LinkToTorchDocCache:
        cache_path = (
            Path(__file__).parent / f"torch_{self.torch_version}_doc_urls.json"
        )
        cache_url = LinkToTorchDocCache(cache_path)
        # Probed namespaces, in priority order. `LinkToTorchDocCache.add`
        # stops at the first hit, so put the most specific / canonical
        # ones first; `torch.Tensor.{}` catches tensor-method ops that
        # don't have a free-function form (e.g. `to_dense`, `index_put`,
        # `masked_scatter`).
        url_tails = (
            "torch.nn.functional.{}.html",
            "torch.{}.html",
            "torch.Tensor.{}.html",
            "torch.linalg.{}.html",
        )
        for a_from_code in rich.progress.track(
            aten_torch_from_code,
            total=len(aten_torch_from_code),
            description=f"Caching torch doc links in '{cache_path.name}'",
        ):
            for tail in url_tails:
                cache_url.add(
                    f"https://docs.pytorch.org/docs/{self.torch_version}"
                    f"/generated/{tail}",
                    a_from_code,
                )
        return cache_url


def print_t(text, file):
    """Print tabbed."""
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


def _md_link(text: str, href: Optional[str]) -> str:
    """Inline-anchor or plain text fallback."""
    if not href:
        return text
    return f'<a href="{href}">{text}</a>'


def _format_aliases(aliases: List[str]) -> str:
    return ", ".join(aliases)


def write_operator_support(
    support_target_name: str,
    support_target_msg: str,
    aten_torch_from_code: List[str],
    supported_opset: Set[str],
    alias_manager: AliasManager,
    official_aten_names: Set[str],
    fh,
    cache_url: LinkToTorchDocCache,
    support_inplace: Set[str],
    support_n_ops_label: str,
):
    """Emit one tabbed section.

    The table is raw HTML (not a markdown pipe table) so each `<tr>`
    can carry a `supported`/`unsupported` class hooked up by the inline
    filter widget at the top of the section.
    """
    row_items: List[tuple] = []
    qte_core = 0
    qte_supported_core = 0
    matched_qte = 0

    print(f'=== "{support_target_name}"', file=fh)
    print("", file=fh)
    for a_from_code in rich.progress.track(
        aten_torch_from_code,
        total=len(aten_torch_from_code),
        description="Generating support table",
    ):
        if alias_manager.is_alias(a_from_code):
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
        alias_str = _format_aliases(alias_manager.get_aliases(a_from_code))
        torch_url_doc = cache_url.get_url(a_from_code)
        op_name_html = _md_link(a_from_code, torch_url_doc)
        row_items.append(
            (
                exist_in_support,
                is_core,
                op_name_html,
                alias_str,
                inplace_str,
                is_core_official_str,
                mapped_in_support_str,
            )
        )

    # Core ops first to keep the historical sort, then unsupported core.
    row_items.sort(key=lambda r: -int(r[1]))

    print_t("", file=fh)
    support_n_ops = len([_ for _ in supported_opset if not _.endswith("_")])
    ratio_total_str = f"{matched_qte}/{len(aten_torch_from_code)}"
    print_t(
        f"Total matched operators in {support_target_msg} compared to:\n\n"
        f"- core PyTorch opset:\n\n"
        f"[={qte_supported_core}/{qte_core} "
        f'"{qte_supported_core}/{qte_core}"]\n\n'
        "-  and support from full `aten::`: \n\n"
        f'[={ratio_total_str} "{ratio_total_str}"]\n\n'
        f" (total operators listed as supported by {support_n_ops_label} "
        f"being {support_n_ops})",
        file=fh,
    )
    print_t("", file=fh)

    # Filter widget + raw HTML table. The filter scope is a single
    # `.op-filter-container`, so multiple sections (TractNNEF, ONNX) on
    # the same page each get their own independent toggle state.
    filter_id = f"op-filter-{support_target_name}"
    print_t(
        '<div class="op-filter-container" markdown="0">\n'
        '<form class="op-filter-form">\n'
        '<label><input type="radio" name="' + filter_id + '" '
        'value="all" checked> All</label>\n'
        '<label><input type="radio" name="' + filter_id + '" '
        'value="supported"> Supported only</label>\n'
        '<label><input type="radio" name="' + filter_id + '" '
        'value="unsupported"> Unsupported only</label>\n'
        "</form>\n"
        '<table class="op-table">\n'
        "<thead><tr>"
        "<th>translated</th><th>aten name</th><th>aliases</th>"
        "<th>can in-place</th><th>is core</th>"
        "</tr></thead>\n"
        "<tbody>",
        file=fh,
    )
    for (
        exist_in_support,
        _is_core,
        op_name_html,
        alias_str,
        inplace_str,
        is_core_official_str,
        mapped_in_support_str,
    ) in row_items:
        klass = "supported" if exist_in_support else "unsupported"
        print_t(
            f'<tr class="op-row {klass}">'
            f"<td>{mapped_in_support_str}</td>"
            f"<td>{op_name_html}</td>"
            f"<td>{alias_str}</td>"
            f"<td>{inplace_str}</td>"
            f"<td>{is_core_official_str}</td>"
            "</tr>",
            file=fh,
        )
    print_t("</tbody>\n</table>\n</div>", file=fh)
    print_t("", file=fh)


FILTER_SCRIPT = """\
<script>
(function () {
  function applyFilter(form) {
    var mode = form.querySelector('input[type="radio"]:checked').value;
    var rows = form.parentElement.querySelectorAll('tr.op-row');
    rows.forEach(function (tr) {
      var sup = tr.classList.contains('supported');
      var keep =
        mode === 'all' ||
        (mode === 'supported' && sup) ||
        (mode === 'unsupported' && !sup);
      tr.style.display = keep ? '' : 'none';
    });
  }
  document.querySelectorAll('.op-filter-form').forEach(function (form) {
    form.addEventListener('change', function () { applyFilter(form); });
  });
})();
</script>
"""


def build_markdown_header(fetcher) -> str:
    date = datetime.now().strftime("%d %b %Y")
    ir_url = fetcher.resolved_ir_url or fetcher.url_ir
    ir_note = (
        f"[PyTorch IR documentation page]({ir_url})"
        if ir_url == fetcher.url_ir
        else (
            f"[PyTorch IR documentation page]({ir_url}) "
            f"(the page for torch {fetcher.torch_version} was emptied "
            f"upstream; falling back to "
            f"torch {LAST_TORCH_VERSION_WITH_IR_DOC} which is the last "
            "version that still enumerates the core ATen IR)"
        )
    )
    return (
        "!!! note\n"
        "    This table and page are auto generated by "
        "`docs/contributing/generate_support_page.py` and reflect the "
        "PyTorch reference docs at the time of generation."
        f" Targeted torch version: **{fetcher.torch_version}**. "
        f"Generated on **{date}**.\n\n"
        "!!! warning\n"
        "     Take these results with a grain of salt: many of the listed "
        "operators never appear in the torch IR graph that "
        "`torch_to_nnef` traces (they get remapped to more generic ops "
        "upstream), and some uncommon operators are rare in real models "
        "so support may be lacking even when marked unsupported. "
        "**SONOS maintains operators on a per-need basis**, "
        "and contributions are always welcome "
        "[see how](./add_new_aten_op.md)."
        "\n\n"
        f"\n 'is core' column refers to this {ir_note}.\n\n"
        "We filter out 'backward' and 'sym' operators from the listing "
        "since they are unwanted in an inference engine. "
        "In-place operations are merged with their non-inplace "
        "counterparts since that distinction is an inference "
        "implementation detail."
    )


def build_markdown_page(torch_version: str):
    """Build supported operators markdown page."""
    fetcher = FetchFromTorchVersion(torch_version)
    official_aten_names, official_prim_names = fetcher.get_core_ir()
    t2n_aten = set(list(aten_ops_registry._registry.keys()))
    onnx_supported, onnx_unsupported = fetcher.get_onnx_support()
    aten_torch_from_code = fetcher.get_aten_torch_from_code()

    aliases_manager = fetcher.get_aliases_from_code()

    support_inplace = set()
    offset = 0
    for ix, a in enumerate(aten_torch_from_code[:]):
        if (  # pylint: disable-next=too-many-boolean-expressions
            a.endswith("_")
            and a[:-1] in aten_torch_from_code
            or aliases_manager.is_alias(a)
            or a.strip() == ""
            or (len(a) and a[0].isupper())
            or "backward" in a
            or a.startswith("sym_")
        ):
            del aten_torch_from_code[ix - offset]
            offset += 1
            support_inplace.add(a[:-1])

    cache_url = fetcher.get_cache_url(aten_torch_from_code)
    with (Path(__file__).parent / "./supported_operators.md").open(
        "w", encoding="utf8"
    ) as fh:
        print(
            build_markdown_header(fetcher),
            file=fh,
        )
        write_operator_support(
            "TractNNEF",
            "`torch_to_nnef`",
            aten_torch_from_code,
            t2n_aten,
            official_aten_names=official_aten_names,
            alias_manager=aliases_manager,
            fh=fh,
            cache_url=cache_url,
            support_inplace=support_inplace,
            support_n_ops_label="`torch_to_nnef`",
        )
        if onnx_supported or onnx_unsupported:
            write_operator_support(
                "ONNX",
                "builtin PyTorch `ONNX` support based on "
                f"[this page]({fetcher.onnx_support_url})",
                aten_torch_from_code,
                onnx_supported,
                official_aten_names=official_aten_names,
                alias_manager=aliases_manager,
                fh=fh,
                cache_url=cache_url,
                support_inplace=support_inplace,
                support_n_ops_label="PyTorch's TorchScript ONNX exporter",
            )
        print(FILTER_SCRIPT, file=fh)
    cache_url.save()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate supported operators markdown page."
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        required=True,
        help="Target PyTorch version to generate the report for.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        assert len(re.findall("\.", args.torch_version)) == 1, (
            "expect X.Y format for torch version"
        )
    assert args.torch_version.replace(".", "").isdigit(), (
        "expect X.Y format for torch version"
    )
    build_markdown_page(torch_version=args.torch_version)
