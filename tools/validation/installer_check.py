"""Exercise the streamed installer, reinstall, and its isolated runtime environment."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f"wheel does not exist: {wheel}")
    with tempfile.TemporaryDirectory(prefix="tensor installer ") as temporary:
        prefix = Path(temporary) / "installation with spaces"
        env = {**os.environ, "TENSOR_INSTALL_DIR": str(prefix),
               "TENSOR_SOURCE": wheel.as_uri(), "TENSOR_EXTRAS": "",
               "TENSOR_NO_MODIFY_PATH": "1"}
        if os.name == "nt":
            shell = shutil.which("pwsh") or shutil.which("powershell")
            command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$input | Out-String | Invoke-Expression"]
            script = (ROOT / "tools/install/install.ps1").read_text()
            executable = prefix / "bin/tensor.exe"
            python = prefix / "tools/tensor-workspace/Scripts/python.exe"
        else:
            command = ["bash"]
            script = (ROOT / "tools/install/install.sh").read_text()
            executable = prefix / "bin/tensor"
            python = prefix / "tools/tensor-workspace/bin/python"
        for _ in range(2):
            subprocess.run(command, input=script, text=True, env=env, check=True, cwd=temporary)
            subprocess.run([str(executable), "--help"], check=True, cwd=temporary, stdout=subprocess.DEVNULL)
        subprocess.run([str(python), "-c", '''
import importlib.abc, importlib.metadata, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tilelang', 'tvm', 'tvm_ffi', 'triton'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
import tensor
assert sys.version_info[:2] == (3, 12)
assert {d.metadata['Name'].lower() for d in importlib.metadata.distributions()} == {'tensor-workspace', 'numpy'}
'''], check=True, cwd=temporary)
        env["TENSOR_EXTRAS"] = "unknown-extra"
        result = subprocess.run(command, input=script, text=True, env=env, cwd=temporary,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            raise RuntimeError("installer accepted an invalid optional dependency")
    print("PASS: streamed install, reinstall, spaces, Python 3.12, compiler-free runtime and invalid options")


if __name__ == "__main__":
    main()
