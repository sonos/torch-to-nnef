"""Fail before pytest if installation changed the Transformers matrix pin."""

import sys
from importlib.metadata import version

expected = sys.argv[1].removeprefix("cli_transformers_").replace("_", ".")
actual = version("transformers")
if actual != expected:
    raise SystemExit(f"Expected transformers=={expected}, installed {actual}")
print(f"Testing transformers=={actual}")
