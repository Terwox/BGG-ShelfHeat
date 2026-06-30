import subprocess
import sys


def test_module_version_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "shelfheat", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "shelfheat 0.1.0"
