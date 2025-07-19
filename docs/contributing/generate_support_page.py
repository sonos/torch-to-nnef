from collections import defaultdict
from datetime import datetime
from pathlib import Path
from torch_to_nnef.op.aten import aten_ops_registry
import requests as rq
import bs4
import subprocess


TORCH_VERSION = "v2.7.1"
URL_IR = "https://docs.pytorch.org/docs/main/torch.compiler_ir.html"
resp = rq.get(URL_IR)

assert resp.status_code == 200

soup = bs4.BeautifulSoup(resp.content, "html.parser")

res = soup.find_all("span", {"class": "pre"})
official_aten_names = set(
    [
        r.text.split(".")[1]
        for r in res
        if r.text.startswith("aten")
        if "backward" not in r.text
    ]
)
official_prim_names = sorted(
    [r.text.split(".")[1] for r in res if r.text.startswith("prim")]
)

t2n_aten = set(list(aten_ops_registry._registry.keys()))

aten_torch_from_code = sorted(
    subprocess.check_output(
        "cd /tmp ; "
        "git -C 'pytorch' pull || git clone -q git@github.com:pytorch/pytorch.git; "
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
    ):
        del aten_torch_from_code[ix - offset]
        offset += 1
        support_inplace.add(a[:-1])


matched_qte = 0
with (Path(__file__).parent / "./supported_operators.md").open(
    "w", encoding="utf8"
) as fh:
    date = datetime.now().strftime("%d %b %Y")
    print(
        "!!! note\n"
        f"    This table and file are auto generated from 'a script' that dig into PyTorch."
        f" Version targetted is:  **'{TORCH_VERSION}'**. file was generated the **{date}**.\n\n"
        "!!! warning\n"
        "     Take these information with a grain of salt as this is referencing operators that may never appear"
        " in torch IR graph traced by `torch_to_nnef` (because remapped to others more generic). Also some "
        " uncommon operators are very rare in models, hence support may be lacking."
        "\n\n"
        f"\n 'is core' column refers to this [pytorch documentation page]({URL_IR})",
        file=fh,
    )
    rows = []
    qte_core = 0
    qte_supported_core = 0
    for a_from_code in aten_torch_from_code:
        if a_from_code in alias_map:
            continue
        is_core = a_from_code in official_aten_names
        is_core_official_str = "✅" if is_core else "-"

        exist_in_t2n = a_from_code in t2n_aten

        if is_core:
            qte_core += 1
            if exist_in_t2n:
                qte_supported_core += 1

        mapped_in_t2n_str = "✅" if exist_in_t2n else "❌"
        if exist_in_t2n:
            matched_qte += 1

        inplace_str = "✅" if a_from_code in support_inplace else "❌"
        alias_str = ", ".join(ref_alias[a_from_code])
        rows.append(
            (
                f"| {a_from_code} | {alias_str} | {inplace_str} | {is_core_official_str} | {mapped_in_t2n_str} |",
                is_core,
            )
        )
    rows = sorted(rows, key=lambda x: -int(x[1]))
    print("", file=fh)
    t2n_n_ops = len([_ for _ in t2n_aten if not _.endswith("_")])
    ratio_total_str = f"{matched_qte}/{len(aten_torch_from_code)}"
    print(
        "Total matched operators in `torch_to_nnef` compared to:\n\n"
        "-  full `aten::`: \n\n"
        f'[={ratio_total_str} "{ratio_total_str}"]\n\n'
        f"- and support from core PyTorch opset:\n\n"
        f'[={qte_supported_core}/{qte_core} "{qte_supported_core}/{qte_core}"]\n\n'
        " (total registered aten "
        f"operators in t2n being {t2n_n_ops})",
        file=fh,
    )
    print("", file=fh)
    print(
        "| aten name | aliases | can in-place | is core | t2n translated |",
        file=fh,
    )
    print(
        "| -------- | ------- | ------- | --------- | ---------------- |",
        file=fh,
    )
    for r in rows:
        print(r[0], file=fh)

    print("", file=fh)
