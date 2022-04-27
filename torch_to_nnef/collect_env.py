""" Used to collect environment status/versions for debuging purpose """
import locale
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from platform import machine

TRACT_PATH = os.environ.get("TRACT_PATH", "tract")


def run_lambda(command):
    """Returns (return-code, stdout, stderr)"""
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    ) as p:
        raw_output, raw_err = p.communicate()
        rc = p.returncode
        if get_platform() == "win32":
            enc = "oem"
        else:
            enc = locale.getpreferredencoding()
        output = raw_output.decode(enc)
        err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def tract_version() -> str:
    return (
        subprocess.check_output(f"{TRACT_PATH} --version".split())
        .decode("utf8")
        .split(maxsplit=1)[-1]
        .strip()
    )


def run_and_read_all(command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def python_version() -> str:
    sys_version = sys.version.replace("\n", " ")
    bits = sys.maxsize.bit_length() + 1
    return f"{sys_version} ({bits}-bit runtime)"


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win32"):
        return "win32"
    if sys.platform.startswith("cygwin"):
        return "cygwin"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def run_and_parse_first_match(command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def check_release_file():
    return run_and_parse_first_match(
        "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )


def get_mac_version():
    return run_and_parse_first_match("sw_vers -productVersion", r"(.*)")


def get_windows_version():
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    wmic_cmd = os.path.join(system_root, "System32", "Wbem", "wmic")
    findstr_cmd = os.path.join(system_root, "System32", "findstr")
    return run_and_read_all(
        f"{wmic_cmd} os get Caption | {findstr_cmd} /v Caption",
    )


def get_lsb_version():
    return run_and_parse_first_match("lsb_release -a", r"Description:\t(.*)")


def get_os() -> str:
    platform_ = get_platform()

    if platform_ in ["win32", "cygwin"]:
        platform_ = get_windows_version()

    if platform_ == "darwin":
        version = get_mac_version()
        if version is None:
            platform_ = "macOS unknown version"
        else:
            platform_ = f"macOS {version} ({machine()})"

    if platform_ == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version()
        if desc is not None:
            platform_ = f"{desc} ({machine()})"
        else:
            # Try reading /etc/*-release
            desc = check_release_file()
            if desc is not None:
                platform_ = f"{desc} ({machine()})"
            else:
                platform_ = f"{platform_} ({machine()})"
    return platform_


def get_pip_packages():
    """Returns `pip list` output. Note: will also find conda-installed pytorch
    and numpy packages."""
    # People generally have `pip` as `pip` or `pip3`
    # But here it is incoved as `python -mpip`
    def run_with_pip(pip):
        if get_platform() == "win32":
            system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
            findstr_cmd = os.path.join(system_root, "System32", "findstr")
            grep_cmd = rf'{findstr_cmd} /R "numpy torch mypy"'
        else:
            grep_cmd = r'grep "torch\|nnef\|numpy"'
        return run_and_read_all(pip + " list --format=freeze | " + grep_cmd)

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip(sys.executable + " -mpip")

    return pip_version, out


def get_gcc_version():
    return run_and_parse_first_match("gcc --version", r"gcc (.*)")


def dump_environment_versions(pathdir: Path):
    with (pathdir / "versions").open("w", encoding="utf8") as fh:
        fh.write(f"tract: {tract_version()}\n")
        fh.write("\n")
        fh.write(f"os: {get_os()}\n")
        fh.write(f"GCC version: {get_gcc_version()}\n")
        fh.write(f"python: {python_version()}\n")
        fh.write(f"python_platform: {platform.platform()}\n")
        fh.write("\n")
        pip_version, pip_output_list = get_pip_packages()
        fh.write(f"Related python package from {pip_version}:\n")
        for line in pip_output_list.split("\n"):
            fh.write(f"{line}\n")
