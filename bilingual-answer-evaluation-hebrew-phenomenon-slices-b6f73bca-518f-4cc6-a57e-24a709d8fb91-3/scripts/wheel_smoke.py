"""Execute the finite smoke from unpacked wheel contents, away from source imports."""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = root / "artifacts"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-check-", dir=scratch) as directory:
        target = Path(directory)
        with zipfile.ZipFile(args.wheel) as wheel:
            # Refuse archive paths outside the temporary extraction root.
            if any(not (target / name).resolve().is_relative_to(target.resolve())
                   for name in wheel.namelist()):
                raise ValueError("Wheel contains an unsafe member path")
            wheel.extractall(target)
        package = target / "cola_lab"
        if not (package / "resources/hebrew_phenomena.json").is_file():
            raise RuntimeError("Wheel is missing the challenge corpus")
        # -c puts cwd first; assert origin so an editable source install cannot mask a broken wheel.
        code = ("from pathlib import Path; import cola_lab; "
                "assert Path(cola_lab.__file__).resolve().parent == Path.cwd() / 'cola_lab'; "
                "from cola_lab.__main__ import main; import sys; "
                "sys.argv = ['cola_lab', 'phenomena-smoke']; main()")
        result = subprocess.run([sys.executable, "-c", code], cwd=target, timeout=60,
                                capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, file=sys.stderr, end="")
            return result.returncode
    print("wheel-smoke: packaged code and corpus executed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
