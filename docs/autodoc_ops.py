"""Write default op NNEF translate docstring to Python file if missing

This is helpfull to allow doc to be properly generated in documentation
website.

"""

import shutil
import inspect
from pathlib import Path

from torch_to_nnef.op.aten import aten_ops_registry


def main():
    for v in set(aten_ops_registry._registry.values()):
        if v._auto_gen_doc:
            file_path = inspect.getfile(v)
            probe = f"def {v.__name__}("
            base_file = Path(file_path)
            bak_file = base_file.parent / base_file.name.replace(
                ".py", ".bak.py"
            )
            shutil.copy(base_file, bak_file)
            signature_line = None
            found_signature = False
            with base_file.open("w") as write_fh:
                with bak_file.open("r") as fh:
                    for line in fh.readlines():
                        write_fh.write(f"{line}")
                        if probe in line:
                            signature_line = True

                        if signature_line and line.endswith(":\n"):
                            found_signature = True
                            write_fh.write(f'    """ {v.__doc__} """\n')
                            signature_line = False
            if found_signature:
                bak_file.unlink()
            else:
                print(v.__name__, "doc addition failed")


if __name__ == "__main__":
    main()
